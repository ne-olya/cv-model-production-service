# Runbook: регрессия качества модели

1. Остановить rollout и сохранить candidate artifact, config и golden-set report.
2. Проверить порядок классов, preprocessing, temperature и artifact SHA-256.
3. Повторно выполнить quality gate на неизменяемом golden set.
4. Сравнить ошибки candidate и production по диагностическим срезам.
5. При подтверждении регрессии вернуть production artifact и открыть разбор причины.

Нельзя менять threshold quality gate только для прохождения конкретного release без согласованного изменения политики качества.
