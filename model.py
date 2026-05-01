import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def apply_log_transform(y):
    return np.log1p(np.maximum(y, 0))

def inverse_log_transform(y_log):
    return np.expm1(y_log)

def get_baseline_scores(X_train, X_test, y_train_log, y_test_original, preprocessor):
    # T1'in rüzgar hızı yokken, komşuların rüzgarıyla T1'i bulmak için daha fazla ağaç (estimators) gerekiyor.
    models = [
        ('lgbm', LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=127, random_state=42, verbose=-1)),
        ('catboost', CatBoostRegressor(n_estimators=2000, learning_rate=0.03, depth=9, random_state=42, verbose=0))
    ]
    
    stacking_model = StackingRegressor(
        estimators=models,
        final_estimator=RidgeCV(),
        passthrough=False,
        cv=3
    )

    results = {}
    trained_pipelines = {}

    for name, model in [*models, ('Stacking', stacking_model)]:
        print(f"🌀 {name} eğitiliyor (Kör Tahmin Modu)...")
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('pre', preprocessor), ('reg', model)])
        
        pipe.fit(X_train, y_train_log)
        y_pred = np.clip(inverse_log_transform(pipe.predict(X_test)), 0, None)
        
        results[name] = {
            'R2': r2_score(y_test_original, y_pred),
            'MAE': mean_absolute_error(y_test_original, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred))
        }
        trained_pipelines[name] = pipe
        print(f"✅ {name} MAE: {results[name]['MAE']:.2f} kW")

    return results, trained_pipelines