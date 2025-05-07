# PIC
Machine Learning / Optimization on portuguese justice operations

models_ipynb - Notebook onde estão criados todos os modelos de ML até agora (arquiteturas individuais, ensembles, etc), e respetivos resultados. 

data_prep.py - Contém uma função data_prep(filename), que pega nos dados do stor e trata-os; retorna os dados tratados prontos para modelação, uma lista dos targets específicos (CC_militar, CC_Execution, etc) e o target total (CC_all)

linear_optimizer.py - Ficheiro python que corre a otimização (com gurobi) para determinado ano, com o modelo escolhido. Modelo de ML tem de ser linear para correspondência com um problema LP. Extração de capacidades dos dados, criação das variáveis de decisão, modelação da função objetivo, das constraints e resultados obtidos.

Ficheiros .joblib são os modelos de ML guardados. Utilizar package joblib para resgatar os modelos noutros files. O nome indica a arquitetura.

Falta o goofy ass romão escrever o que faz cada file e organizar, a partir de agora trabalhamos por aqui para termos tudo sempre updated.

João sabes o caminho ou queres que eu te diga?
