# classifier/views.py (full)
import os, time, json
import matplotlib
matplotlib.use("Agg")   # Must be before pyplot import
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from PIL import Image
import numpy as np

from .models import Prediction
from .forms import ImageUploadForm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, InceptionV3, VGG16, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# -------------------
# Helper utilities
# -------------------
def remove_corrupted_images(folder_path):
    removed = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                Image.open(file_path).verify()
            except:
                try:
                    os.remove(file_path)
                    removed += 1
                except:
                    pass
    return removed

def load_dataset():
    dataset_path = os.path.join(settings.MEDIA_ROOT, "datasets")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train = datagen.flow_from_directory(
        dataset_path, target_size=(224,224), batch_size=16,
        class_mode="categorical", subset="training", shuffle=True
    )
    val = datagen.flow_from_directory(
        dataset_path, target_size=(224,224), batch_size=16,
        class_mode="categorical", subset="validation", shuffle=True
    )
    return train, val

# -------------------
# Landing Page
# -------------------
def landing_page(request):
    total_users = User.objects.count()
    total_predictions = Prediction.objects.count()
    return render(request, "landing.html", {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "now": time.localtime()
    })

# -------------------
# Admin auth & dashboard
# -------------------
def admin_login(request):
    msg = ""
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        # simple admin check (you can use real admin users)
        if username == "admin" and password == "admin":
            request.session['admin'] = username
            return redirect('admin_dashboard')
        else:
            msg = "Invalid credentials"
    return render(request, "classifier/admin_login.html", {"msg": msg})

def admin_logout(request):
    request.session.pop('admin', None)
    return redirect('landing_page')

def admin_dashboard(request):
    if 'admin' not in request.session:
        return redirect('admin_login')
    return render(request, "classifier/admin_dashboard.html")

# -------------------
# Upload & View Dataset
# -------------------
def upload_dataset(request):
    if 'admin' not in request.session:
        return redirect('admin_login')

    message = ""
    class_counts = {}
    graph_url = None
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'datasets')
    os.makedirs(dataset_path, exist_ok=True)

    if request.method == "POST" and request.FILES.get('dataset'):
        dataset_file = request.FILES['dataset']
        fs = FileSystemStorage(location=dataset_path)
        filename = fs.save(dataset_file.name, dataset_file)

        # if zip -> extract
        if filename.endswith(".zip"):
            import zipfile
            zip_ref = zipfile.ZipFile(os.path.join(dataset_path, filename), 'r')
            zip_ref.extractall(dataset_path)
            zip_ref.close()
            os.remove(os.path.join(dataset_path, filename))

        message = "Dataset uploaded/extracted successfully!"

    # count classes
    if os.path.exists(dataset_path):
        for folder in sorted(os.listdir(dataset_path)):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                class_counts[folder] = len(files)

    if class_counts:
        plt.figure(figsize=(10,4))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title("Dataset Distribution")
        plt.xticks(rotation=45)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, "graphs"), exist_ok=True)
        graph_path = os.path.join(settings.MEDIA_ROOT, "graphs/dataset_graph.png")
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
        graph_url = settings.MEDIA_URL + "graphs/dataset_graph.png"

    return render(request, "classifier/upload_dataset.html", {
        "message": message, "class_counts": class_counts, "graph_url": graph_url
    })


# -------------------
# Preprocess Page
# -------------------
def preprocess_dataset(request):
    if 'admin' not in request.session:
        return redirect('admin_login')
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'datasets')
    if not os.path.exists(dataset_path):
        return render(request, "classifier/preprocess.html", {"error":"Dataset not uploaded."})

    image_sizes = []
    corrupted = 0
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                try:
                    with Image.open(img_path) as im:
                        image_sizes.append(im.size)
                except:
                    corrupted += 1

    widths = [w for w,h in image_sizes]
    heights = [h for w,h in image_sizes]

    if image_sizes:
        plt.figure(figsize=(8,4))
        plt.hist(widths, bins=20, alpha=0.7, label='Widths')
        plt.hist(heights, bins=20, alpha=0.7, label='Heights')
        plt.legend()
        graph_path = os.path.join(settings.MEDIA_ROOT, "graphs/preprocess_graph.png")
        plt.tight_layout(); plt.savefig(graph_path); plt.close()
        graph_url = settings.MEDIA_URL + "graphs/preprocess_graph.png"
    else:
        graph_url = None

    return render(request, "classifier/preprocess.html", {
        "total_images": len(image_sizes),
        "corrupted": corrupted, "graph_url": graph_url
    })


def load_dataset():
    dataset_path = os.path.join(settings.MEDIA_ROOT, "datasets")

    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True,
        shear_range=0.2,
        brightness_range=[0.7,1.3]
    )

    train = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train, val


# -------------------
# TRAINING - multiple models
# -------------------
def train_model_generic(request, model_name):
    if 'admin' not in request.session:
        return redirect('admin_login')

    os.makedirs(os.path.join(settings.MEDIA_ROOT, "graphs"), exist_ok=True)

    # Load dataset
    train, val = load_dataset()
    class_indices = train.class_indices

    # Select model
    if model_name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
        save_name = "mobilenet"
    elif model_name == "inception":
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224,3))
        save_name = "inception"
    elif model_name == "vgg":
        base = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
        save_name = "vgg16"
    elif model_name == "resnet":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
        save_name = "resnet50"
    else:
        return render(request, "classifier/train_result.html", {"error": "Invalid model name"})

    for layer in base.layers[:-50]:
      layer.trainable = False

    for layer in base.layers[-50:]:
     layer.trainable = True
    # Build model
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    output = Dense(len(class_indices), activation='softmax')(x)

    model = Model(base.input, output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    start = time.time()
    history = model.fit(train, validation_data=val, epochs=10,verbose=1)
    end = time.time()

    # Save model
    model_path = os.path.join(settings.MEDIA_ROOT, f"{save_name}.keras")
    model.save(model_path)

    # Save class labels
    with open(os.path.join(settings.MEDIA_ROOT, f"{save_name}_classes.json"), "w") as f:
        json.dump(class_indices, f)

    # ========== SAVE COMPARISON DATA ==========
    comparison_file = os.path.join(settings.MEDIA_ROOT, "comparison_data.json")

    # Prepare new accuracy record
    acc_record = {
        "model": save_name,
        "val_accuracy": float(max(history.history["val_accuracy"]))
    }

    # Load old data or initialize
    if os.path.exists(comparison_file):
        try:
            data = json.load(open(comparison_file))
        except:
            data = []
    else:
        data = []

    # Remove old entry of same model
    data = [d for d in data if d["model"] != save_name]

    # Add new entry
    data.append(acc_record)

    # Save file
    json.dump(data, open(comparison_file, "w"), indent=4)
    # ===========================================

    # Graph paths
    acc_path = os.path.join(settings.MEDIA_ROOT, f"graphs/{save_name}_accuracy.png")
    loss_path = os.path.join(settings.MEDIA_ROOT, f"graphs/{save_name}_loss.png")

    # Accuracy graph
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title(f"{save_name.upper()} Accuracy")
    plt.savefig(acc_path)
    plt.close()

    # Loss graph
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title(f"{save_name.upper()} Loss")
    plt.savefig(loss_path)
    plt.close()

    return render(request, "classifier/train_result.html", {
        "model_name": save_name.upper(),
        "accuracy_graph": settings.MEDIA_URL + f"graphs/{save_name}_accuracy.png",
        "loss_graph": settings.MEDIA_URL + f"graphs/{save_name}_loss.png",
        "train_acc": round(history.history["accuracy"][-1], 4),
        "val_acc": round(history.history["val_accuracy"][-1], 4),
        "train_time": round(end - start, 2)
    })


def train_mobilenet(request):
    return train_model_generic(request, "mobilenet")

def train_inception(request):
    return train_model_generic(request, "inception")

def train_vgg(request):
    return train_model_generic(request, "vgg")

def train_resnet(request):
    return train_model_generic(request, "resnet")


# -------------------
# Comparison (simple)
# -------------------
def comparison(request):
    if 'admin' not in request.session:
        return redirect('admin_login')

    comparison_file = os.path.join(settings.MEDIA_ROOT, "comparison_data.json")

    if not os.path.exists(comparison_file):
        return render(request, "classifier/comparison.html", {
            "error": "Train at least one model to generate comparison graph."
        })

    # Load data
    data = json.load(open(comparison_file))

    model_names = [d["model"].upper() for d in data]
    accuracies = [d["val_accuracy"] for d in data]

    # Generate graph
    graph_path = os.path.join(settings.MEDIA_ROOT, "graphs/comparison_accuracy.png")

    plt.figure(figsize=(8,5))
    bars = plt.bar(model_names, accuracies)
    plt.xlabel("Model")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison - Validation Accuracy")

    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{acc:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return render(request, "classifier/comparison.html", {
        "graph_url": settings.MEDIA_URL + "graphs/comparison_accuracy.png",
        "data": data
    })


# -------------------
# Users details
# -------------------
def users_details(request):
    if 'admin' not in request.session:
        return redirect('admin_login')
    users = User.objects.all()
    user_data = []
    for u in users:
        count = Prediction.objects.filter(user=u).count()
        user_data.append({"username":u.username, "email":u.email, "date_joined":u.date_joined, "predictions":count})
    return render(request, "classifier/users_details.html", {"users": user_data})

# -------------------
# User auth & dashboard
# -------------------
def register_user(request):
    message = ""
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        if User.objects.filter(username=username).exists():
            message = "Username exists"
        else:
            User.objects.create_user(username=username, email=email, password=password)
            message = "Registered. Please login."
    return render(request, "user/register.html", {"message": message})

def login_user(request):
    message = ""
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('user_dashboard')
        else:
            message = "Invalid credentials"
    return render(request, "user/login.html", {"message": message})

@login_required(login_url='/login')
def user_dashboard(request):
    user = request.user
    raw_preds = Prediction.objects.filter(user=user).order_by('-created_at')[:10]
    preds = []
    for p in raw_preds:
        try:
            if p.image and os.path.exists(p.image.path):
                preds.append(p)
        except:
            pass
    return render(request, "user/user_dashboard.html", {"user": user, "predictions": preds})

def user_logout(request):
    logout(request)
    return redirect('landing_page')

# -------------------
# Prediction page
# -------------------
def predict_disease(request):
    if not request.user.is_authenticated:
        return redirect('login_user')

    prediction_label = None
    confidence_percent = None
    image_url = None

    model_path = os.path.join(settings.MEDIA_ROOT, "mobilenet.keras")
    if not os.path.exists(model_path):
        return render(request, "user/predict.html", {
            "error": "Model not found. Admin must train model."
        })

    model = load_model(model_path)

    class_file = os.path.join(settings.MEDIA_ROOT, "mobilenet_classes.json")
    if not os.path.exists(class_file):
        return render(request, "user/predict.html", {
            "error": "Class mapping not found."
        })

    with open(class_file, "r") as f:
        class_indices = json.load(f)

    # Reverse mapping
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    if request.method == "POST" and request.FILES.get("image"):

        uploaded_file = request.FILES["image"]

        save_dir = os.path.join(settings.MEDIA_ROOT, "user_predictions")
        os.makedirs(save_dir, exist_ok=True)

        fs = FileSystemStorage(location=save_dir)
        filename = fs.save(uploaded_file.name, uploaded_file)

        saved_path = os.path.join(save_dir, filename)
        image_url = settings.MEDIA_URL + "user_predictions/" + filename

        # Preprocess image
        img = Image.open(saved_path).convert("RGB")
        img = img.resize((224, 224))

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Prediction
        preds = model.predict(arr)

        pred_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        # Convert to %
        confidence_percent = round(confidence * 100, 2)

        prediction_label = idx_to_class.get(pred_idx, "Unknown")

        # Save prediction
        Prediction.objects.create(
            user=request.user,
            image="user_predictions/" + filename,
            predicted_class=prediction_label,
            confidence=confidence_percent
        )

    return render(request, "user/predict.html", {
        "prediction_label": prediction_label,
        "confidence": confidence_percent,
        "image_url": image_url
    })