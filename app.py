import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import asyncio
import aiohttp
import vk_config
import os
import io
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logger = logging.getLogger(__name__)

# Настройка пути загрузки
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Инициализация моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Используется устройство: {device}")
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Функция для получения ID города
async def get_city_id(session, city_name):
    search_url = "https://api.vk.com/method/database.getCities"
    params = {
        "access_token": vk_config.ACCESS_TOKEN,
        "v": vk_config.VERSION,
        "country_id": 1,  # Россия
        "q": city_name,
        "count": 1
    }
    try:
        async with session.get(search_url, params=params) as response:
            data = await response.json()
            if "response" in data and data["response"]["items"]:
                city_id = data["response"]["items"][0]["id"]
                logger.info(f"ID города '{city_name}': {city_id}")
                return city_id
            else:
                logger.warning(f"Город '{city_name}' не найден.")
    except Exception as e:
        logger.exception(f"Ошибка при получении ID города '{city_name}': {e}")
    return None

# Функция для поиска пользователей по параметрам
async def search_users_by_params(session, city_name=None, name=None):
    city_id = None
    if city_name:
        city_id = await get_city_id(session, city_name)
    
    search_url = "https://api.vk.com/method/users.search"
    params = {
        "access_token": vk_config.ACCESS_TOKEN,
        "v": vk_config.VERSION,
        "q": name if name else '',
        "city": city_id if city_id else '',
        "count": 10,
        "fields": "photo_50,photo_100,photo_200"
    }

    try:
        async with session.get(search_url, params=params) as response:
            data = await response.json()
        logger.info(f"Получен ответ от VK API: {data}")
    except Exception as e:
        logger.exception(f"Ошибка при поиске пользователей: {e}")
        return []

    user_results = []
    if "response" in data:
        for item in data["response"]["items"]:
            user_results.append({
                "user_id": item["id"],
                "name": f"{item['first_name']} {item['last_name']}",
                "photo": item.get("photo_200", item.get("photo_100", item.get("photo_50", "")))
            })
        logger.info(f"Найдено пользователей: {len(user_results)}")
    else:
        if "error" in data:
            error = data["error"]
            logger.error(f"Ошибка VK API {error['error_code']}: {error['error_msg']}")
    return user_results

# Асинхронная функция для получения фотографий из альбомов
async def get_photos_from_albums(session, user_id, face_embedding, user_name, default_photo_url):
    logger.info(f"Получение фотографий для пользователя {user_id} ({user_name}) из альбомов")
    vk_results = []
    search_url = "https://api.vk.com/method/photos.getAll"
    params = {
        "access_token": vk_config.ACCESS_TOKEN,
        "v": vk_config.VERSION,
        "owner_id": user_id,
        "count": 100
    }
    
    try:
        async with session.get(search_url, params=params) as response:
            data = await response.json()
        logger.info(f"Получен ответ от VK API для {user_id}: {data}")
    except Exception as e:
        logger.exception(f"Ошибка при получении фотографий для пользователя {user_id}: {e}")
        return {"profile_url": f"https://vk.com/id{user_id}", "name": user_name, "error": "Ошибка при получении фотографий", "photo_url": default_photo_url}
    
    if "response" in data:
        for item in data["response"]["items"]:
            photo_url = item["sizes"][-1]["url"]

            try:
                async with session.get(photo_url) as photo_response:
                    photo_content = await photo_response.read()
                img = Image.open(io.BytesIO(photo_content)).convert('RGB')
                img_cropped_list, _ = mtcnn(img, return_prob=True)
                if img_cropped_list is not None:
                    embeddings = [resnet(img_cropped.unsqueeze(0).to(device)) for img_cropped in img_cropped_list]
                    
                    for embedding in embeddings:
                        distance = (face_embedding - embedding).norm().item()
                        if distance < 0.6:  # Порог расстояния для определения совпадения
                            vk_results.append({
                                "name": user_name,
                                "profile_url": f"https://vk.com/id{user_id}",
                                "photo_url": photo_url
                            })
                            break
            except Exception as e:
                logger.exception(f"Ошибка при обработке фотографии для пользователя {user_id}: {e}")
    else:
        if "error" in data and data["error"]["error_code"] == 30:  # Ошибка доступа к закрытому профилю
            logger.warning(f"Профиль пользователя {user_id} закрыт")
            return {"profile_url": f"https://vk.com/id{user_id}", "name": user_name, "error": "Профиль закрыт", "photo_url": default_photo_url}
        else:
            logger.error(f"Не удалось получить фотографии для пользователя {user_id}")
            return {"profile_url": f"https://vk.com/id{user_id}", "name": user_name, "error": "Не удалось получить фотографии", "photo_url": default_photo_url}
    
    return vk_results

# Обработчик главной страницы
@app.route('/', methods=['GET', 'POST'])
async def index():
    logger.info("Запрос на главную страницу")
    if request.method == 'POST':
        logger.info("Обработка POST запроса")
        if 'file' not in request.files:
            logger.warning("Нет файла в запросе")
            return render_template('index.html', error='Нет файла в запросе')
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Файл не выбран")
            return render_template('index.html', error='Файл не выбран')
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                logger.info(f"Файл сохранен по пути: {filepath}")
            except Exception as e:
                logger.exception(f"Ошибка при сохранении файла: {e}")
                return render_template('index.html', error='Ошибка при сохранении файла')
    
            city = request.form.get('city')
            name = request.form.get('name')
    
            try:
                logger.info(f"Загрузка изображения: {filepath}")
                img = Image.open(filepath).convert('RGB')
                img_cropped_list, _ = mtcnn(img, return_prob=True)
                if img_cropped_list is not None:
                    face_encodings = [resnet(img_cropped.unsqueeze(0).to(device)).detach() for img_cropped in img_cropped_list]
                    logger.info(f"Лиц обнаружено: {len(face_encodings)}")
                else:
                    logger.info("Лица не найдены")
                    return render_template('index.html', error='Лица не найдены', filename=filename)
            except Exception as e:
                logger.exception(f"Ошибка при загрузке изображения: {e}")
                return render_template('index.html', error=f"Ошибка при загрузке изображения: {e}", filename=filename)
    
            vk_results = []
            try:
                async with aiohttp.ClientSession() as session:
                    user_results = await search_users_by_params(session, city_name=city, name=name)
                    if not user_results:
                        logger.info("Не найдено пользователей для данных параметров")
                        return render_template('index.html', error='Не найдено пользователей для данных параметров', filename=filename)
                    
                    tasks = [
                        get_photos_from_albums(session, user["user_id"], face_encoding, user["name"], user["photo"])
                        for user in user_results
                        for face_encoding in face_encodings
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, dict) and 'error' in result:
                            vk_results.append(result)
                        elif isinstance(result, list):
                            vk_results.extend(result)
                        elif isinstance(result, Exception):
                            logger.exception("Ошибка при обработке задачи:", exc_info=result)
                    if not vk_results:
                        logger.info("Совпадающих профилей не найдено")
                        return render_template('index.html', error='Совпадающих профилей не найдено', filename=filename)
            except Exception as e:
                logger.exception(f"Ошибка при поиске профилей: {e}")
                return render_template('index.html', error=f"Ошибка при поиске профилей: {e}", filename=filename)
    
            return render_template('index.html', filename=filename, vk_results=vk_results)
    
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Запуск сервера Flask")
    app.run(debug=True)
