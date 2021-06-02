import pandas as pd

maad_face = pd.read_pickle("/Users/ntdat/Downloads/MAAD_Face_1.0.pkl")
maad_face.to_numpy()

print(maad_face)
