
# Project Ideas
## Deep Neural Networks
### **Convolutional Neural Networks**
1. Transform into 3 images
2. (C x T) => (3 x 150)
3. xy, xz, yz and show the temporal evolution => apply CNN (sparse representation)

### **Fully Connected**
#### Similar idea but with fully connected networks
1. flatten all => try to predict
2. 3 x 150 => serve each coordinate to a single perceptron
3. flatten all => go through latent space (create embeddings) => predict over latent space
4. 150 x 3 => serve each datapoint to a single perceptron => 150 perceptrons on the first layer

## Classical Machine Learning Algorithms
### K-Nearest Neighbours
### Random Forests
### SVM
### Naive Bayes

---
## Data Processing
1. normalizare x, y, z peste dimensiunea temporala => (3,) medii, peste tot datasetu
2. normalizare x, y, z separat pentru fiecare moment in parte => (3 x 150) medii, peste tot datasetu