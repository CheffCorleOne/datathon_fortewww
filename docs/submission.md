# Сдача решения (раздел 8 ТЗ)

| Что требуется | Где |
|---|---|
| Адрес сервиса с интерфейсом | `https://<адрес>/` — выдаётся `tunnel.ps1`; актуальный адрес передаётся жюри отдельно |
| Адрес API | `POST https://<адрес>/api/v1/analyze`, `GET https://<адрес>/api/v1/jobs/{job_id}` |
| API-ключ | Заголовок `X-API-Key`; значение `API_KEY` из `.env` передаётся жюри отдельно, в репозиторий не попадает |
| Инструкция по запуску и использованию | [README.md](../README.md) |
| Архитектура, подход, модели | [architecture.md](architecture.md) |
| Нормативные источники и сопоставление правил с видео | [rules.md](rules.md) |
| Перечень распознаваемых нарушений | [violations.md](violations.md) |
| Ограничения и известные ошибки | [limitations.md](limitations.md) |

## Проверка API за 30 секунд

```bash
curl -X POST https://<адрес>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
# {"job_id":"3f2a9c1e0b7d","status":"processing"}
curl https://<адрес>/api/v1/jobs/3f2a9c1e0b7d -H "X-API-Key: <ключ>"
# {"job_id":"...","status":"done","verdict":"violation","violations":[{"type":"...","timecode_sec":19.5,
#   "description":"...","evidence_frames":[{"timestamp_sec":19.5,"image_base64":"/9j/..."}], ...}]}
```

Без ключа или с неверным ключом сервис отвечает `401`. Пока видео обрабатывается, ответ — `{"status":"processing","stage":"...","progress":0.4}`. При ошибке декодирования — `{"status":"failed","error":"..."}`.

## Результаты на открытом наборе

40 уникальных видео (42 файла, два — побайтовые дубликаты), прогон через HTTP API: `tools/eval.py --api`.
Итоговые цифры и таблица по видео — в [README.md](../README.md#оценка-на-обучающем-наборе).
