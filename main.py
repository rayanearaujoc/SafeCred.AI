import sweetviz as sv
import pandas as pd

df = pd.read_csv("bank_transactions_data_2.csv") 
relatorio = sv.analyze(df)
relatorio.show_html("relatorio_fraude.html")  