# import funkcí
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split # pro 2. způsob rozdělení dat na trénování a testování
import pandas as pd

# načtení dat
df = pd.read_csv("Coimbra_breast_cancer_dataset.csv")

# rozdělení dat na X, y
# X = všechny data kromě sloupce "Classification", y = pouze sloupec "Classification"
# V tomto případě X jsou informace o pacientovi, y říka jestli konkrétní pacient měl rakovinu prsu nebo ne
X, y = df.drop('Classification', axis=1), df[['Classification']].values.ravel()

# Určení dat na trénování a testování - 1. způsob
n_train = int(df.shape[0]*2/3)

X_train = X[:n_train]
X_test = X[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]

# 2. způsob - Ve většině případech je tento způsob lepší, protože data i zamíchá. V mém případě toto není potřeba, protože data nejsou nijak seřazená.
#X_train, X_test, y_train, y_test = train_test_split(X,y)

# Umělá inteligence
# Nevím proč její přesnost je v tomto případě tak nízká. Nejspíš tyto data nejsou pro tuto umělou inteligenci moc užitečná.
ai = GaussianNB() 
ai.fit(X_train, y_train) # učení AI
score = ai.score(X_test, y_test) # vyzkoušení AI na nových datech, hodnocení přesnosti
print(f"Přesnost: {score*100}%") # vypsání přesnosti