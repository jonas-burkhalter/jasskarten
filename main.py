from roboflow import Roboflow

## https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/model/2
#rf = Roboflow(api_key="twAd4Wqzp9zwpscja84N")
#project = rf.workspace().project("playing-cards-ow27d")
#model = project.version(2).model

## https://universe.roboflow.com/augmented-startups/playing-cards-kwt8k/model/1
rf = Roboflow(api_key="twAd4Wqzp9zwpscja84N")
project = rf.workspace().project("playing-cards-kwt8k")
model = project.version(1).model

# infer on a local image
print(model.predict("poker5.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())