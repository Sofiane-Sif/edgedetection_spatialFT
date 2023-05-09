"""
Created on Fri Apr 30 20:43:08 2021
@title : Introduction à la segmentation d'image par contours
@author: Sofiane SIFAOUI - TIPE TF
"""

from matplotlib import pyplot as plt
import numpy as np

#Importation de la donnée brute image de papillon monarque

img = plt.imread('papillon.jpg') 
plt.imshow(img)
plt.title("Donnée brute : Image d'un papillon monarque")
plt.axis('off') 
plt.show()

img=img[:,:,0] #on convertit l'image uniquement suivant 2 dimensions
print(img.shape)


#ETAPE 1 : Application de la transformée de Fourier en 2 dimensions à l'image


f = np.fft.fft2(img)
print(f[0][1])

#On réarrange le spectre de fréquences obtenu grace à la fonction fftshift qui
# permet le centrage de la fréquence nulle au centre de l'image. 

f_centre = np.fft.fftshift(f)
print(f_centre[207][310])

#Obtention du spectre d'amplitude de l'image : 
    
spectre_amplitude = 20*np.log(abs(f_centre))

#Affichage, position du problème : 
    
plt.figure(figsize=(12, 12)) 
plt.subplot(1,2,1) 
plt.imshow(img, cmap = 'gray') #on souhaite afficher l'image en niveau de gris
plt.title('Image à étudier')
plt.subplot(1,2,2)
plt.imshow(spectre_amplitude, cmap = 'gray')  
plt.title(r"Spectre d'amplitude : Espace des fréquences spatiales (de Fourier)")
plt.show()



#=============ETAPE 2 : Création d'un filtre passe-haut. Nous allons bloquer les basses fréquences pour obtenir, au final, uniquement les contours de l'image de base===========



lignes, colonnes = img.shape 
ligne_au_centre, colonne_au_centre = int(lignes/2), int(colonnes/2)

masque = np.ones((lignes, colonnes, 2), np.uint8)
masque=masque[:,:,0] 
r = 105 #Rayon du masque
centre = [ligne_au_centre, colonne_au_centre]
x, y = np.ogrid[:lignes, :colonnes] 
aire_masque = (x - centre[0])**2 + (y - centre[1])**2 <= r*r #equation d'un cercle de centre (207,310)
masque[aire_masque] = 0 #Toutes les valeurs à l'intérieur du cercle valent 0. 
#Autrement dit, toutes les basses fréquences ont été supprimées par le filtre.

#Application du filtre passe-haut : 
    
fonction_filtre = f_centre * masque 
fonction_filtre_amplitude = 20 * np.log(np.abs(fonction_filtre)) #Amplitude de la fonction filtrée




#=============ETAPE 3 : Retour dans le domaine spatial et obtention de l'image segmenté suivant les contours de l'image de référence===========


f_decentre = np.fft.ifftshift(fonction_filtre) 
image_segmente = np.fft.ifft2(f_decentre) #TFinverse en 2 dimensions
image_segmente = np.abs(image_segmente) 




#=============ETAPE 4 : Visualisation. Affichage des 4 graphiques sur une même zone===========
    
fig = plt.figure(figsize=(15, 15)) 
ax1 = fig.add_subplot(2,2,1) 
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Image de référence : Papillon')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(spectre_amplitude, cmap='gray')
ax2.title.set_text(r"Spectre d'amplitude de la TF de l'image, obtenu par FFT")
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(fonction_filtre_amplitude, cmap='gray')
ax3.title.set_text(r"Spectre d'amplitude + Filtre passe haut")
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(image_segmente, cmap='gray')
ax4.title.set_text("Filtre de détéction des contours de l'image du papillon, obtenu par IFFT")
plt.show()


#CONCLUSION :  

#En résumée, nous avons converti notre donnée brute (l'image d'un papillon) dans l'espace des fréquences spatiales (espace de Fourier) afin de lui appliquer un filtre passe-haut, 
# dans le but de représenter uniquement ses contours. Cet exemple d'application de la TF au service du traitement d'image constitue une première introduction à la segmentation d'images, en illustrant notamment la technique
#de segmentation par détection des contours (edge-based segmentation). Ainsi, en parvenant à segmenter l'image suivant les contours des motifs des ailes du papillon, nous pourrons analyser, puis classifier les images de papillon qui survivent le mieux dans
#un milieu donné. On sera alors en mesure de détecter, en fonction des motifs présents sur les ailes du papillon, quelle espèce s'adaptera et survivera le mieux dans tel ou tel milieu. Mieux encore, nous pourrons, au fur et à mesure des exemples 
#traitées par l'ordinateur, faire "apprendre" à l'ordinateur. Ce-dernier pourra alors être en mesure, en fonction de l'image qui lui est donné à étudier, de dire si tel ou tel papillon sera avantagé dans tel ou tel milieu : on appelle cela l'apprentissage automatique des ordinateurs, ou encore le machine learning.


