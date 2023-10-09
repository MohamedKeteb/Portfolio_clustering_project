## 1ère séance 

1ère question : quelle distance pour la matrice de similarité ? comment définir les poids ?
2ème question : faire varier k 
3ème question : comment marche le package signet 
4ème question : quels algorithmes de clusterings appliquer à la matrice de similarité ? 
5ème question : comment tenir compte du temps qui passe 
6ème question : comment fusionner les micro-portefeuilles 

Idée : 
- Les stocks redondants dans un cluster, i.e. ceux dont la fréquence d’apparition est la plus élevée
sont les plus représentatifs ils auront plus de poids
- On caractérise un cluster par les 5 (arbitraire) actifs les plus représentatifs de celui-ci.
Si on a K clusters, on se retrouve avec 5K actifs (sous reserve que 5K soit plus petit que N = nombre total d’actifs)
et on fait Markowitz dessus. 
