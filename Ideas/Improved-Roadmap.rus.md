# Error Modeling

### 0\. Разбивка: добавляем честный тест

Сделаем трёхуровневую схему:

1.  `base_dataset` делим на:
    *   **70% — meta\_train** (для обучения всех моделей и построения loss),
    *   **30% — meta\_test** (честная проверка ансамбля, к нему вообще не прикасаемся, пока всё не обучено).

```python
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

Дальше работаем **только с meta\_train**.

* * *

### 1\. Строим loss на meta\_train через OOF

Считаем ошибку для всех 70% посредством OOF:

1.  На `meta_train` делаем `KFold(n_splits=5, shuffle=True, random_state=42)`.
2.  В каждом фолде:
    *   обучаем `best_model` на 4 фолдах,
    *   предсказываем на оставшемся фолде,
3.  Собираем OOF-предсказания `y_pred_oof` для всех строк `meta_train`.
4.  Считаем `loss` для каждой строки:
    ```python
    loss = np.abs(y_meta_train - y_pred_oof)
    ```
5.  Делаем `loss_dataset`:
    *   признаки: `X_meta_train` (+ можно добавить `y_pred_oof` как отдельный фича),
    *   таргет: `loss`.

Так вы получите **14k точек с честной оценкой ошибки**.

* * *

### 2\. Обучаем error\_predict\_model

Теперь:

*   `error_predict_model` обучаем на `loss_dataset` (cross-validation внутри при желании).
*   Это именно обучение на meta\_train, а meta\_test к этому не привлекаем.

Важно: на meta\_train всё честно, потому что для каждой строки loss считался по модели, которая **не видела** эту строку при обучении (OOF).

* * *

### 3\. Обучаем alternative\_model

Здесь варианты:

*   самый простой: обучить `alternative_model` на **всём meta\_train** (как обычную модель цены),
*   чуть более «тематический» вариант: обучить `alternative_model` только на «трудных» примерах из meta\_train (по реальному loss), но это уже вторая итерация.

На первом шаге достаточно обучить её так же, как `best_model`, только, возможно, с другой архитектурой/классом.

* * *

### 4\. Тестируем идею ансамбля на meta\_test

Теперь берём **meta\_test** (те самые 30%, которых модели ещё не видели).

1.  Обучаем финальную версию `best_model` на **всём meta\_train**.
2.  Обучаем финальную версию `error_predict_model` на всём `loss_dataset` (внутри можно было настроить гиперпараметры).
3.  Обучаем финальную `alternative_model` на meta\_train.

Теперь на `meta_test`:

*   считаем `y_pred_best = best_model.predict(X_meta_test)`,
*   считаем `y_pred_alt = alternative_model.predict(X_meta_test)`,
*   считаем `predicted_loss = error_predict_model.predict(X_meta_test)`.

Дальше:

*   выбираем порог (например, top-30% по `predicted_loss`),
*   для строк с `predicted_loss` выше порога берём `y_pred_alt`,  
    для остальных – `y_pred_best`,
*   по этим «смешанным» предсказаниям считаем итоговую метрику (R², MAE, MAPE, что вам удобнее),
*   сравниваем с «голой» `best_model` на этом же meta\_test.

Если ансамбль даёт выигрыш → идея реально работает.

* * *

