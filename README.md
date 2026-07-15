# VisionServe: эксплуатация CV-модели

VisionServe — самостоятельный production-контур для модели классификации изображений. Он отвечает за API-контракт, валидацию, batch inference, версию модели, контейнеризацию, тестирование, структурные логи и базовый мониторинг. Исследовательское обучение намеренно остаётся в отдельном репозитории EcoSort.

Дополнительные эксплуатационные каталоги:

- `load_testing/` — k6-профиль с порогами error rate и p95 latency;
- `runbooks/` — инструкции для инцидентов высокой latency и регрессии модели.

## API

| Метод | Endpoint | Назначение |
|---|---|---|
| `POST` | `/predict` | один JPEG/PNG/WebP в multipart-поле `file` |
| `POST` | `/predict-batch` | от 1 до 16 файлов в полях `files` |
| `GET` | `/health` | readiness процесса и модели |
| `GET` | `/model-info` | версия, backend, устройство, классы и размер входа |
| `GET` | `/metrics` | метрики в Prometheus text format |
| `GET` | `/monitoring-summary` | JSON для простого dashboard |

Ошибочные MIME type, пустые, повреждённые и слишком большие файлы отвергаются до инференса. В ответе есть label, confidence, все вероятности, версия и latency.

TorchScript artifact проходит fail-fast проверку SHA-256 sidecar, порядка классов и формы logits на warm-up запросе. `X-Request-ID` возвращается клиенту и связывает API-ответ со структурным логом.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,monitoring]'
uvicorn app.main:app --reload
```

```bash
curl -F 'file=@example.jpg' http://localhost:8000/predict
curl http://localhost:8000/model-info
pytest -q
python scripts/plot_monitoring.py --api http://127.0.0.1:8000
```

Без `VISION_MODEL_PATH` сервис использует детерминированный `color-smoke` backend. Он нужен только для проверки API, Docker и CI и не является ML-моделью. Для реального сервиса экспортируйте EcoSort в TorchScript, положите файл в `models/` и задайте переменные из `.env.example`. TorchScript должен принимать нормированный tensor `N×3×224×224` и возвращать logits `N×7` в порядке `VISION_CLASSES`.

## Docker

```bash
docker compose up --build
```

API: `http://localhost:8000`, Swagger: `http://localhost:8000/docs`, dashboard: `http://localhost:8501`. Контейнер запускается непривилегированным пользователем и имеет healthcheck. Для GPU создайте override с NVIDIA runtime и `VISION_DEVICE=cuda`; базовый образ оставлен CPU-only и воспроизводимым.

## Тесты и CI/CD

Тесты покрывают декодирование/нормализацию, загрузку backend, single/batch API, неверный тип файла и полный smoke inference. `.gitlab-ci.yml` последовательно выполняет lint, unit tests, сборку образа и health smoke test внутри контейнера.

Job `model-quality-gate` сравнивает отчёты кандидата и production по macro F1, допустимой регрессии и latency. Отчёт кандидата должен быть заново вычислен на неизменяемом golden set; metadata из произвольного checkpoint не считается доказательством качества.

## Логи и мониторинг

Сервис пишет JSON-события запроса и предсказания: request ID, endpoint, latency, версию, класс и confidence; содержимое изображения не логируется. In-memory метрики показывают число запросов/ошибок, среднюю latency/confidence и распределение классов. Это достаточно для demo, но сбрасывается при рестарте и не агрегируется между workers.

Для production подключите Prometheus/OpenTelemetry, histogram для p50/p95/p99, централизованные логи, alerts по error rate/latency и persistent dashboard. Drift отслеживайте по входным embeddings и распределению confidence с эталонным окном; изменение долей классов само по себе не доказывает drift.

## Надёжность и безопасность

Перед внешним доступом нужны reverse proxy, TLS, authentication, rate limiting, timeout, антивирусная/decoder sandbox проверка, ограничение concurrency и очистка временных файлов. Для нескольких workers модель загрузится в память каждого процесса. Версия Docker-образа, model artifact hash, список классов и preprocessing должны выпускаться как единая совместимая версия.
