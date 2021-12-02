import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

PCA_model = PCA(n_components=2)


def get_domain_MMD(path):
    array = np.load(path)
    reducted_array = PCA_model.fit_transform(array)
    return reducted_array


def get_domain_Original(path):
    array = np.load(path)
    array = np.transpose(array)
    array = PCA_model.fit_transform(array)
    return array


Art = get_domain_Original("data_legacy/new_Art.npy")
Art_MMD = get_domain_MMD("./model/Arts/Art_MMD.npy")
Art_Toy_MMD = get_domain_MMD("./model/Arts/Art_Toy_MMD.npy")

CD = get_domain_Original("data_legacy/CD.npy")
CD_MMD = get_domain_MMD("./model/CD/CD_MMD.npy")
CD_Toy_MMD = get_domain_MMD("./model/CD/CD_Toy_MMD.npy")

Digital = get_domain_Original("data_legacy/Digital.npy")
Digital_MMD = get_domain_MMD("./model/Digital/Digital_MMD.npy")
Digital_Toy_MMD = get_domain_MMD("./model/Digital/Digital_Toy_MMD.npy")

E = get_domain_Original("data_legacy/new_E.npy")
E_MMD = get_domain_MMD("./model/Electronic/E_MMD.npy")
E_Toy_MMD = get_domain_MMD("./model/Electronic/E_Toy_MMD.npy")

Kindle = get_domain_Original("data_legacy/Kindle.npy")
Kindle_MMD = get_domain_MMD("./model/Kindle/Kindle_MMD.npy")
Kindle_Toy_MMD = get_domain_MMD("./model/Kindle/Kindle_Toy_MMD.npy")

Movie = get_domain_Original("data_legacy/Movie.npy")
Movie_MMD = get_domain_MMD("./model/Movie/Movie_MMD.npy")
Movie_Toy_MMD = get_domain_MMD("./model/Movie/Movie_Toy_MMD.npy")

Video = get_domain_Original("data_legacy/Video.npy")
Video_MMD = get_domain_MMD("./model/Video/Video_MMD.npy")
Video_Toy_MMD = get_domain_MMD("./model/Video/Video_Toy_MMD.npy")

domains = ["Arts", "CD", "Digital", "Electronic", "Kindle", "Movie", "Video"]
sources = [Art, CD, Digital, E, Kindle, Movie, Video]
MMDs = [Art_MMD, CD_MMD, Digital_MMD, E_MMD, Kindle_MMD, Movie_MMD, Video_MMD]
target_MMDs = [Art_Toy_MMD, CD_Toy_MMD, Digital_Toy_MMD, E_Toy_MMD, Kindle_Toy_MMD, Movie_Toy_MMD, Video_Toy_MMD]
colors = ['#B8860B', '#98FB98', '#FF4500', "#9370DB", "#00FFFF", "#808080", "#FF1493", "#0000CD"]

Toy = get_domain_Original("data_legacy/new_Toy.npy")
Toy_MMD = 0
for item in target_MMDs:
    Toy_MMD += item
Toy_MMD = Toy_MMD / len(target_MMDs)
print(Toy_MMD.shape)

plt.subplot(1, 2, 1)
plt.scatter(Art[:, 1], Art[:, 0], s=1, c='#B8860B', label='Arts', alpha=0.8)
plt.scatter(CD[:, 1], CD[:, 0], s=1, c='#98FB98', label='CD', alpha=0.7)
plt.scatter(Digital[:, 1], Digital[:, 0], s=1, c='#FF4500', label='Digital', alpha=0.6)
plt.scatter(E[:, 1], E[:, 0], s=1, c='#9370DB', label='Electronic', alpha=0.5)
plt.scatter(Kindle[:, 1], Kindle[:, 0], s=1, c='#00FFFF', label='Kindle', alpha=0.4)
plt.scatter(Movie[:, 1], Movie[:, 0], s=1, c='#808080', label='Movie', alpha=0.3)
plt.scatter(Video[:, 1], Video[:, 0], s=1, c='#FF1493', label='Videoy', alpha=0.2)
plt.scatter(Toy[:, 1], Toy[:, 0], s=1, c='#0000CD', label='Toy (Target Domain)')
plt.grid(linestyle='-.')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(Art_MMD[:, 1], Art_MMD[:, 0], s=1, c='#B8860B', label='Arts->Toy', alpha=0.8)
plt.scatter(CD_MMD[:, 1], CD_MMD[:, 0], s=1, c='#98FB98', label='CD->Toy', alpha=0.7)
plt.scatter(Digital_MMD[:, 1], Digital_MMD[:, 0], s=1, c='#FF4500', label='Digital->Toy', alpha=0.6)
plt.scatter(E_MMD[:, 1], E_MMD[:, 0], s=1, c='#9370DB', label='Electronic->Toy', alpha=0.5)
plt.scatter(Kindle_MMD[:, 1], Kindle_MMD[:, 0], s=1, c='#00FFFF', label='Kindle->Toy', alpha=0.4)
plt.scatter(Movie_MMD[:, 1], Movie_MMD[:, 0], s=1, c='#808080', label='Movie->Toy', alpha=0.3)
plt.scatter(Video_MMD[:, 1], Video_MMD[:, 0], s=1, c='#FF1493', label='Video->Toy', alpha=0.2)
plt.scatter(Toy_MMD[:, 1], Toy_MMD[:, 0], s=1, c='#0000CD', label='Toy (Target Domain)')
plt.grid(linestyle='-.')
plt.legend()

plt.show()
