import sweetviz as sv
import pandas as pd

df = pd.read_csv("creditcard.csv") 
relatorio = sv.analyze(df)
relatorio.show_html("relatorio_fraude.html")  
 
