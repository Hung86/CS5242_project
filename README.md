# CS5242_project

### Meeting Minutes (03 OCT 2020)

- Everyone will build their own models and compare the results with each other
- Make sure to record down important steps
- Next meeting 17 OCT 2020!

22 OCT 2020
Traning experience: 
- no globalPooling -> overfitting
- GlobalMaxPooling -> slow training, low accuracy and validation accuracy
- GlobalAvgPooling -> good result
- Training strategy : Cross validation, data augmentation with ImageDataGenerator (Kerras), label smoothing, dropout, bacthnormalization
- VGG (0.91), resnet (0.91), xception (0.95)
- image preprocessing : added random noise to image (not work)

