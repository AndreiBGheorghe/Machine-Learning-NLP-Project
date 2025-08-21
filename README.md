[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rhtjUFgW)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=19555951&assignment_repo_type=AssignmentRepo)
# Predicția complexității cuvintelor 2025

Link de invite pentru competiția de pe kaggle: https://www.kaggle.com/t/488080b161694d58a0c66d5c07eeffa6

## Cuprins

- [Ce este complexitatea unui cuvânt?](#complex)
- [Caracteristici](#features)
- [Cod de start](#start)

<a name="complex"></a> 
## Ce este complexitatea unui cuvânt?

Complexitatea unui cuvânt este un criteriu subiectiv și depinde de mulți factori de la cât de des este întâlnit în vorbire acel cuvânt, cât de lung sau greu de citit este, dacă este un termen specializat, forma morfologică, semantică până la funcția cuvântului în sintaxa propoziției.  Pe baza acestor idei, putem să ne definim niște funcții care să extragă diverse caracterisitici.

<a name="features"></a> 
## Caracteristici
Caracteristicile pentru un cuvânt sunt sub forma unui vector:
$$\mathbf{x} = \{ x_1, \cdots, x_n \}$$

unde fiecare componentă $x_i$ e o valoare numerică care poate fi extrasă prin euristici:

- cu privire la cuvânt: frecvență într-un corpus de dimensiuni mari, frecvență într-un corpus de texte pentru copii, lungime, nr de silabe, nr de consoane, dacă e abreviere
- cu privire la sintaxă: parte de vorbire, funcția în propozție
- caracteristici psiholingvistice: vârsta de achiziție, imaginabilitatea cuvântului
- caracteristici semantice: număr de sinonime, antonime, cuvinte similare din alte propoziții
- caracteristici binare: prezența în anumite liste externe, dacă e de tip titlu, dacă este entitate numită etc
 
Pentru fiecare astfel de vector vrem să prezicem o valoare numerică y care reprezintă scorul de complexitate.
Scorul de complexitate depinde de mulți factori, nu doar de proprietățile individuale ale cuvântului, ci și de contextul în care este folosit, funcția și sensurile pe care le are în contextul respectiv.

<a name="start"></a> 
## Cod de start
În repo aveți câteva notebooks de la care puteți porni. Acestea pot fi uploadate și pe kaggle și pot rula direct acolo (dar trebuie să elimnați prima celulă care downloadează datele).

- baseline.ipynb, [colab](https://colab.research.google.com/drive/1AhovRCbL5jyLEzAS9TFgu7igILJF6dCm?usp=sharing) - vă arată un proces de extragere a caracteristicilor din text; cu câteva trucuri acest baseline vă poate da nota 6
- spacy*.ipynb, [colab](https://colab.research.google.com/drive/1sbnw6BfOoneyD_gFWNCov_MN5KDVk-DM?usp=sharing) - tutorial spacy și proprietățile pe care le puteți folosi 
- transformer*.ipynb, [colab](https://colab.research.google.com/drive/1zAyi_b-OoaxoR0Bnq0Ec2-rKj1h3aXtn?usp=sharing) - tutorial huggingface transformers, cum să obții reprezentări din ultimul strat din rețea