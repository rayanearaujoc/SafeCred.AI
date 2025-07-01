import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModeloEvaluator:
    def __init__(self, csv_path):
        print("🔍 Lendo CSV...")
        self.df = pd.read_csv(csv_path)
        print("✅ CSV carregado.")
        self._preprocessar()
        self._splitar()

    def _preprocessar(self):
        print("🧪 Pré-processando dados...")
        scaler = StandardScaler()
        self.df['Amount'] = scaler.fit_transform(self.df[['Amount']])
        self.df.drop(columns=['Time'], inplace=True, errors='ignore')
        print("✅ Pré-processamento concluído.")

    def _splitar(self):
        print("🔀 Separando treino e teste...")
        X = self.df.drop(columns=['Class'])
        y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("✅ Split concluído.")

    def avaliar_modelos(self):
        print("🚀 Avaliando modelos...")
        ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        modelos = {
            "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
            "Random Forest": RandomForestClassifier(class_weight='balanced'),
            "XGBoost": XGBClassifier(scale_pos_weight=ratio, use_label_encoder=False, eval_metric='logloss'),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }

        resultados = []

        for nome, modelo in modelos.items():
            print(f"🔄 Treinando e avaliando: {nome}...")
            modelo.fit(self.X_train, self.y_train)
            y_pred = modelo.predict(self.X_test)

            try:
                y_proba = modelo.predict_proba(self.X_test)[:, 1]
                roc_auc = roc_auc_score(self.y_test, y_proba)
            except:
                roc_auc = None

            resultados.append({
                "Modelo": nome,
                "Accuracy": round(accuracy_score(self.y_test, y_pred), 6),
                "Precision": round(precision_score(self.y_test, y_pred), 6),
                "Recall": round(recall_score(self.y_test, y_pred), 6),
                "F1 Score": round(f1_score(self.y_test, y_pred), 6),
                "ROC AUC": round(roc_auc, 6) if roc_auc is not None else None
            })
            print(f"✅ {nome} avaliado.")

        self.df_resultados = pd.DataFrame(resultados)
        print("✅ Todos os modelos avaliados.")
        return self.df_resultados

