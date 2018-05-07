import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import itertools
import torch
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

def layer_outputs(image):
    outputs = []
    names = []
    for layer in modulelist[1:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    output_im = []
    for i in outputs:
        i = i.squeeze(0)
        temp = to_grayscale(i)
        output_im.append(temp.data.cpu().numpy())

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (30, 50)


    for i in range(9):
        a = fig.add_subplot(8,4,i+1)
        try:
            imgplot = plt.imshow(output_im[i])
            plt.axis('off')
            a.set_title(names[i].partition('(')[0], fontsize=30)
        except:
            pass

    plt.savefig('layer_outputs.jpg', bbox_inches='tight')

def load_image(path):
    image = Image.open(path)
    return image

def normalize(image):
    normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
    )
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])
    image = Variable(preprocess(image).unsqueeze(0).cuda())
    return image

def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image

def plot_x(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_first_layer(image_path, layer):
    x = load_image(image_path)
    x = normalize(x)
    x = layer(x)
    x = x.squeeze(0)
    plt.subplots(figsize=(10,10))
    for i in range(6):
        y = F.relu(x[i])
        plt.subplot(2, 3, i+1)
        #plt.tight_layout()
        #plt.imshow(temp2.data.cpu().numpy(), vmin=0, vmax=10)
        plt.imshow(y.data.cpu().numpy(), vmin = 0, vmax = 10, cmap='gnuplot2')
        plt.colorbar()
        title = "{:.2e}".format(np.abs(y.data.cpu().numpy().sum()))
        plt.title(title,fontsize=20)

def layers_by_classification(valid_path, class_names, net, index):
    layer = list(net.modules())[index]
    layer_activation_strength = [defaultdict(float) for _ in range(layer.out_channels)]
    strongest_activator = []


    val_image_paths = [[valid_path.format(class_name) + f for f in listdir(valid_path.format(class_name)) if isfile(join(valid_path.format(class_name), f))] for class_name in class_names]
    val_image_paths = [item for sublist in val_image_paths for item in sublist]
    val_image_classes = [[class_name for f in listdir(valid_path.format(class_name)) if isfile(join(valid_path.format(class_name), f))] for class_name in class_names]
    val_image_classes  = [item for sublist in val_image_classes for item in sublist]

    #print(val_image_paths)
    for image_path, image_class in zip(val_image_paths, val_image_classes):
        #print(image_path)

        x = load_image(image_path)
        x = normalize(x)
        x = layer(x)
        x = x.squeeze(0)

        for i in range(len(x)):
            y = F.relu(x[i])
            layer_activation_strength[i][image_class] += y.data.cpu().numpy().sum()
            #activation_strength()

    for i in range(layer.out_channels):
        denominator = sum(layer_activation_strength[i].values())
        for key in layer_activation_strength[i].keys():
            layer_activation_strength[i][key] /= denominator

        strongest_activator.append(max(layer_activation_strength[i], key=layer_activation_strength[i].get))

    return layer_activation_strength, strongest_activator

def plot_first_layer_with_strongest_activator(image_path, layer, strongest_activator, net, class_names):
    predicted_class, probability = predict_one_image(image_path, class_names, net)

    plt.subplots(figsize=(10,10))
    x = load_image(image_path)
    crop = transforms.RandomResizedCrop(224,scale =(0.5,1.0))
    x = crop(x)
    #x = normalize(x)
    plt.subplot(2, 4, 1)
    plt.imshow(x)
    #plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.axis('off')
    plt.title('original image\nP({}): {:.2f}'.format(predicted_class, probability), fontsize = 15)
    x = normalize(x)
    x = layer(x)
    x = x.squeeze(0)
    #plt.suptitle('P({}): {:.2f}'.format(predicted_class, probability), fontsize = 30)

    for i in range(layer.out_channels):
        y = F.relu(x[i])
        plt.subplot(2, 4, i+2)
        plt.axis('off')
        #plt.imshow(temp2.data.cpu().numpy(), vmin=0, vmax=10)
        plt.imshow(y.data.cpu().numpy(), vmin = 0, vmax = 5, cmap='gnuplot2')
        #plt.colorbar()
        title = '{} layer'.format(strongest_activator[i])
        title += "\n{:.2}".format(np.abs(y.data.cpu().numpy().mean()))

        plt.title(title,fontsize=15)

    plt.subplots_adjust(wspace=0.1, hspace=0.01)
    #plt.tight_layout()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def predict_one_image(image_path, class_names, net):
    x = load_image(image_path)
    x = normalize(x)
    outputs = net(x)
    probability, predicted = torch.max(outputs.data,1)
    probability = probability.exp().tolist()[0]

    return class_names[predicted[0]], probability

def prediction_list(loader, net):
    dataiter = iter(loader)
    ground_truth = []
    prediction = []
    probability_list = []

    for images, labels in dataiter:

        ground_truth += list(labels)
        outputs = net(Variable(images.cuda()))
        probability, predicted = torch.max(outputs.data, 1)
        prediction += list(predicted)
        probability_list += probability.exp().tolist()

    return ground_truth, probability_list, prediction

def predict_proba_for_hidden_class(loader, net):
    dataiter = iter(loader)
    prediction = []
    probability_list = []

    for images, labels in dataiter:

        outputs = net(Variable(images.cuda()))
        probability, predicted = torch.max(outputs.data, 1)
        prediction += list(predicted)
        probability, predicted = torch.max(outputs.data,1)
        #print(probability)
        probability_list += probability.exp().tolist()


    return probability_list, prediction

def plot_confusion_matrix_hidden_class(cm, classes, hidden_class_prediction, hidden_class_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                                      ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    hidden_class_confusion = [hidden_class_prediction.count(i) for i in range(3)]
    cm = np.append(cm, [np.array(hidden_class_confusion)], axis = 0)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    y_tick_marks = np.arange(len(classes)+1)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(y_tick_marks, classes + [hidden_class_name])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix_extra_label(cm, classes, extra_x_label, extra_y_label,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes)+1)
    plt.xticks(tick_marks, classes + [extra_x_label], rotation=45)
    plt.yticks(tick_marks, classes + [extra_y_label])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def accuracy_vs_confidence_plot(ground_truth, probability_list_base_classes, prediction_base_classes, probability_list_outlier, prediction_outlier, class_names):

    prediction_data = []
    for confidence_threshold in np.linspace(0,0.999,21):

        adjusted_pred_dj = [pred if proba > confidence_threshold else len(class_names) for proba, pred in zip(probability_list_outlier, prediction_outlier)]
        adjusted_pred_base_classes = [pred if proba > confidence_threshold else len(class_names) for proba, pred in zip(probability_list_base_classes, prediction_base_classes)]
        cnf_matrix_adjusted = confusion_matrix(ground_truth, adjusted_pred_base_classes)
        hidden_class_confusion = [adjusted_pred_dj.count(i) for i in range(4)]

        if cnf_matrix_adjusted.shape[1] == len(class_names):
            cnf_matrix_adjusted = np.append(cnf_matrix_adjusted, np.matrix(np.zeros(cnf_matrix_adjusted.shape[0])).T, axis = 1)
            cnf_matrix_adjusted = np.append(cnf_matrix_adjusted, [hidden_class_confusion], axis = 0)
        else:
            cnf_matrix_adjusted[-1:] = hidden_class_confusion
            cnf_matrix_adjusted = np.matrix(cnf_matrix_adjusted)

        outlier_total = cnf_matrix_adjusted[-1:].sum()
        outlier_false_positive_rate = cnf_matrix_adjusted[-1:, :-1].sum()/outlier_total
        base_class_predicted_totals = cnf_matrix_adjusted[:-1].sum(axis = 0)
        base_class_predicted_totals = base_class_predicted_totals.tolist()[0]

        base_class_actual_totals = cnf_matrix_adjusted[:-1].sum(axis = 1)
        base_class_precision = [cnf_matrix_adjusted[i,i]/base_class_predicted_totals[i] for i in range(len(class_names))]
        base_class_recall = [cnf_matrix_adjusted[i,i]/base_class_actual_totals[i] for i in range(len(class_names))]
        prediction_data.append([confidence_threshold, outlier_false_positive_rate, np.mean(base_class_recall), np.mean(base_class_precision)])

    confidence_threshold = [item[0] for item in prediction_data]
    outlier_fp_rate = [item[1] for item in prediction_data]
    base_class_recall = [item[2] for item in prediction_data]
    base_class_precision = [item[3] for item in prediction_data]

    plt.plot(confidence_threshold, outlier_fp_rate)
    plt.plot(confidence_threshold, base_class_recall)
    plt.plot(confidence_threshold, base_class_precision)
    plt.legend(['outlier false-positive rate', 'avg base class recall', 'avg base class precision'])
    plt.xlabel('required confidence level')
    plt.ylabel('proportion')

    plt.show()


def plot_training_data(data_path):
    fig, big_axes = plt.subplots( figsize=(20, 20) , nrows=3, ncols=1, sharey=True)
    base_class_names = listdir(f'{data_path}train')

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(base_class_names[row-1], fontsize=16)
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top=False, bottom=False, left=False, right=False)
        big_ax._frameon = False

    count = 1
    for class_ in base_class_names:
        for i in range(5):
            ax = fig.add_subplot(3,5,count)
            file = listdir(f'{data_path}train/{class_}')[i]
            img = plt.imread(f'{data_path}train/{class_}/{file}')
            ax.imshow(img);
            ax.set_aspect('equal')
            ax.axis('off')
            count +=1

    fig.set_facecolor('w')
    fig.subplots_adjust(wspace=0.1, hspace=0.11)
    plt.show()

def get_data_loaders_init(data_dir):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
        'dwayne_johnson': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
        }


    image_datasets = {x: datasets.ImageFolder(join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'dwayne_johnson']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'valid', 'dwayne_johnson']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'dwayne_johnson']}
    class_names = image_datasets['train'].classes

    return class_names, dataloaders

def plot_validation_data_with_prediction(data_path, class_names, net, n_images):
    fig, big_axes = plt.subplots( figsize=(20, 20) , nrows=3, ncols=1, sharey=True)
    base_class_names = listdir(f'{data_path}train')

    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title('True class: {}'.format(base_class_names[row-1]), fontsize=16)
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top=False, bottom=False, left=False, right=False)
        big_ax._frameon = False

    count = 1
    for class_ in base_class_names:
        for i in range(n_images):

            ax = fig.add_subplot(3,n_images,count)
            file = listdir(f'{data_path}valid/{class_}')[i]
            img = plt.imread(f'{data_path}valid/{class_}/{file}')
            predicted_class, probability = predict_one_image(f'{data_path}valid/{class_}/{file}', class_names, net)

            ax.imshow(img);
            ax.set_aspect('equal')
            ax.axis('off')
            #ax.set_title(file)
            ax.set_title('Predicted: {}'.format(predicted_class))
            count +=1

    fig.set_facecolor('w')
    fig.subplots_adjust(wspace=0.1, hspace=0.11)
    plt.show()

def get_data_loaders(data_dir):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
        }


    image_datasets = {x: datasets.ImageFolder(join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    return class_names, dataloaders


def get_net_accuracy_stats(dataloaders, net, epoch):
    validloader = dataloaders['valid']
    rockloader = dataloaders['dwayne_johnson']
    ground_truth, probability_list_base_classes, prediction = prediction_list(validloader, net)
    base_class_prediction_confidence = [np.mean([proba for proba, pred in zip(probability_list_base_classes, prediction) if pred == i]) for i in range(len(set(prediction)))]
    cnf_matrix = confusion_matrix(ground_truth, prediction)
    precisions = [row[i]/row.sum() for i, row in enumerate(cnf_matrix)]
    total_accuracy = np.sum([truth == pred for truth, pred in zip(ground_truth, prediction)])/len(ground_truth)
    probability_list_dj, prediction = predict_proba_for_hidden_class(rockloader, net)


    return([epoch+1, precisions, total_accuracy, base_class_prediction_confidence, np.mean(probability_list_dj)])


def plot_training_stats(training_stats, class_names):
    x = [row[0] for row in training_stats]
    class_precisions = [row[1] for row in training_stats]
    total_accuracy = [row[2] for row in training_stats]
    base_class_prediction_confidence = [row[3] for row in training_stats]
    dj_confidence = [row[4] for row in training_stats]
    fig = plt.figure(figsize = (13,6))
    plt.plot(x, base_class_prediction_confidence)
    plt.plot(x, dj_confidence, 'red')
    #plt.plot(x, total_accuracy, 'red', linewidth=3.0)
    plt.legend(class_names + ['DJ'])
    plt.ylabel('confidence of prediction')
    plt.xlabel('epoch')
    plt.title('Average prediction confidence vs. training epoch')
    plt.show()

def plot_first_layer_with_strongest_activator_resnet(image_path, layer, strongest_activator, net, class_names):
    plt.subplots(figsize=(10,40))
    x = load_image(image_path)
    crop = transforms.RandomResizedCrop(224,scale =(0.5,1.0))
    x = crop(x)
    #x = normalize(x)
    plt.subplot(13, 5, 1)
    predicted_class, probability = predict_one_image_resnet(image_path, class_names, net)
    plt.imshow(x)
    #plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.axis('off')
    plt.title('original image\nP({}): {:.2f}'.format(predicted_class, probability), fontsize = 15)
    x = normalize(x)
    x = layer(x)
    x = x.squeeze(0)
    #plt.suptitle('P({}): {:.2f}'.format(predicted_class, probability), fontsize = 30)

    for i in range(layer.out_channels):
        y = F.relu(x[i])
        plt.subplot(13, 5, i+2)
        plt.axis('off')
        #plt.imshow(temp2.data.cpu().numpy(), vmin=0, vmax=10)
        plt.imshow(y.data.cpu().numpy(), vmin = 0, vmax = 5, cmap='gnuplot2')
        #plt.colorbar()
        title = '{} layer'.format(strongest_activator[i])
        title += "\n{:.2}".format(np.abs(y.data.cpu().numpy().mean()))

        plt.title(title,fontsize=15)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.tight_layout()

def prediction_list_resnet34(loader, net):
    dataiter = iter(loader)
    ground_truth = []
    prediction = []
    probability_list = []

    for images, labels in dataiter:

        ground_truth += list(labels)
        outputs = net(Variable(images.cuda()))
        normalized_data = F.softmax(Variable(outputs.data), 1)
        probability, predicted = torch.max(normalized_data.data, 1)
        prediction += list(predicted)

        #print(probability)
        probability_list += probability.tolist()


    return ground_truth, probability_list, prediction

def predict_one_image_resnet(image_path, class_names, net):

    x = load_image(image_path)
    x = normalize(x)
    outputs = net(x)

    outputs = F.softmax(Variable(outputs.data), 1)
    probability, predicted = torch.max(outputs.data,1)
    probability = probability.tolist()[0]

    return predicted.tolist()[0], probability

def predict_proba_for_hidden_class_resnet34(loader, net):
    dataiter = iter(loader)
    prediction = []
    probability_list = []

    for images, labels in dataiter:

        outputs = net(Variable(images.cuda()))
        outputs = F.softmax(Variable(outputs.data), 1)
        probability, predicted = torch.max(outputs.data, 1)
        prediction += list(predicted)
        #print(probability)
        probability_list += probability.tolist()


    return probability_list, prediction
