# CNN-project

Main_training.py est la pour lancer l'apprentisage des differents réseaux de neurone. Ses réseaux doivent être crée avec Convolution_Layer.py.
Convolution_layer_cpu.py et Convolution_Layer.py sont similaire, seul la façon d'apprentisage change: Convolution_layer_cpu.py utilise concurrent.futures pour faire du multi-processing.
Convolution_layer_gpu.py est une tentative d'utilisation du gpu pour les calcules matriciel a l'aide de cupy.

Les images d'aprentisage doivent être stocker dans un dossier (train ou train_64 dans mon cas, non donné ici) et indexé dans un fichier texte avec leurs categories. De même pour les images de validation.

PS: les commentaires du code(quand ils sont présents) sont abominables: mélange d'anglais/français et fautes d'orthographes. Désolé.
