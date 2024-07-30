import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


def load_model(model_path):

    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model


def process_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    return image


def create_data_batches(x, batch_size=32):
    """Create a batched dataset from image paths."""
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch


loaded_full_model = load_model("./Models/20240530-08041717056280-1000-images.h5")


def get_pred_label(prediction_probabilities):
    unique_breeds = [
        "affenpinscher",
        "afghan_hound",
        "african_hunting_dog",
        "airedale",
        "american_staffordshire_terrier",
        "appenzeller",
        "australian_terrier",
        "basenji",
        "basset",
        "beagle",
        "bedlington_terrier",
        "bernese_mountain_dog",
        "black-and-tan_coonhound",
        "blenheim_spaniel",
        "bloodhound",
        "bluetick",
        "border_collie",
        "border_terrier",
        "borzoi",
        "boston_bull",
        "bouvier_des_flandres",
        "boxer",
        "brabancon_griffon",
        "briard",
        "brittany_spaniel",
        "bull_mastiff",
        "cairn",
        "cardigan",
        "chesapeake_bay_retriever",
        "chihuahua",
        "chow",
        "clumber",
        "cocker_spaniel",
        "collie",
        "curly-coated_retriever",
        "dandie_dinmont",
        "dhole",
        "dingo",
        "doberman",
        "english_foxhound",
        "english_setter",
        "english_springer",
        "entlebucher",
        "eskimo_dog",
        "flat-coated_retriever",
        "french_bulldog",
        "german_shepherd",
        "german_short-haired_pointer",
        "giant_schnauzer",
        "golden_retriever",
        "gordon_setter",
        "great_dane",
        "great_pyrenees",
        "greater_swiss_mountain_dog",
        "groenendael",
        "ibizan_hound",
        "irish_setter",
        "irish_terrier",
        "irish_water_spaniel",
        "irish_wolfhound",
        "italian_greyhound",
        "japanese_spaniel",
        "keeshond",
        "kelpie",
        "kerry_blue_terrier",
        "komondor",
        "kuvasz",
        "labrador_retriever",
        "lakeland_terrier",
        "leonberg",
        "lhasa",
        "malamute",
        "malinois",
        "maltese_dog",
        "mexican_hairless",
        "miniature_pinscher",
        "miniature_poodle",
        "miniature_schnauzer",
        "newfoundland",
        "norfolk_terrier",
        "norwegian_elkhound",
        "norwich_terrier",
        "old_english_sheepdog",
        "otterhound",
        "papillon",
        "pekinese",
        "pembroke",
        "pomeranian",
        "pug",
        "redbone",
        "rhodesian_ridgeback",
        "rottweiler",
        "saint_bernard",
        "saluki",
        "samoyed",
        "schipperke",
        "scotch_terrier",
        "scottish_deerhound",
        "sealyham_terrier",
        "shetland_sheepdog",
        "shih-tzu",
        "siberian_husky",
        "silky_terrier",
        "soft-coated_wheaten_terrier",
        "staffordshire_bullterrier",
        "standard_poodle",
        "standard_schnauzer",
        "sussex_spaniel",
        "tibetan_mastiff",
        "tibetan_terrier",
        "toy_poodle",
        "toy_terrier",
        "vizsla",
        "walker_hound",
        "weimaraner",
        "welsh_springer_spaniel",
        "west_highland_white_terrier",
        "whippet",
        "wire-haired_fox_terrier",
        "yorkshire_terrier",
    ]
    return unique_breeds[np.argmax(prediction_probabilities)]


def main():
    st.title("Dog Breed Predictor")

    st.write("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a dog image...", type=["jpg", "png", "jpeg"]
    )

    upload_folder = "./uploads"

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    if uploaded_file is not None:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded and saved successfully!")

    uploaded_images = [
        f for f in os.listdir(upload_folder) if f.endswith(("jpg", "png", "jpeg"))
    ]
    selected_image = st.selectbox(
        "Select an image from the uploads folder:", uploaded_images, index=None
    )

    if st.button("Run"):
        if selected_image:
            with st.status("PREDECTING", expanded=True, state="running"):
                image = ["./uploads/" + selected_image]
                image_batch = create_data_batches(image)
                prediction = loaded_full_model.predict(image_batch)
                label = get_pred_label(prediction)
                st.header(label)
        else:
            st.warning("SELECT AN IMAGE", icon="🚨")


if __name__ == "__main__":
    main()
