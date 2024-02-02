import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomResizedCrop,
    RandomHorizontalFlip, ColorJitter
)
import warnings
warnings.filterwarnings("ignore")

def compute_hyperplans_and_grid(feats):
    """
        Given features, compute the hyperplans between each class and returns the a grid of predictions.
        - In:
            * feats : Feature tensor with lower dimension (Here dim=2).
            * args : args parser object.
            * crops : Handle multiple crops per image. For now set it to False
        - Out:
            * plot_grid : A dictionnary with necessary
    """
    assert feats.shape[-1]==2, 'Error, features should be of dimension 2'
    grid_size=100
    x_min, x_max = feats.min(dim=0)[0][0].item(), feats.max(dim=0)[0][0].item()
    y_min, y_max = feats.min(dim=0)[0][1].item(), feats.max(dim=0)[0][1].item()

    xmingrid = x_min-abs(x_min)*0.4
    ymingrid = y_min-abs(y_min)*0.4
    xmaxgrid = x_max+abs(x_max)*0.4
    ymaxgrid = y_max+abs(y_max)*0.4

    x = torch.linspace(xmingrid, xmaxgrid, grid_size)  # extent of the grid on the x axis
    y = torch.linspace(ymingrid, ymaxgrid, grid_size)  # extent of the grid on the y axis
    [xx, yy] = torch.meshgrid(x, y)
    grid_points = torch.stack([xx.ravel(), yy.ravel()], axis=1).to(feats.device)
    distances = torch.norm((grid_points.reshape(-1,1,2)-feats.reshape(1,-1,2)), dim=2, p=2)
    Z = torch.min(distances, dim = 1)[1]
    Z = Z.reshape(grid_size, grid_size)

    hyperplans = []
    for n in range(-1, 2):
        x1 = feats[n].cpu() # centroid class 2
        x2 = feats[n+1].cpu() # centroid class 2
        middle = (x1+x2)/2
        a = (x1[1]-x2[1])/(x1[0]-x2[0])
        b = middle[1] + middle[0]/a
        if abs(a)<10e-3:
            hyperplans.append([np.inf,middle[0], middle])
        elif abs(a)>10e3:
            hyperplans.append([0,middle[1], middle])
        else:
            hyperplans.append([-1/a.item(),b.item(), middle])

    def get_intersect(line1, line2):
        a1, b1, _ = line1
        a2, b2, _ = line2
        x = (b2-b1)/(a1-a2)
        y = a1*(b2-b1)/(a1-a2) + b1
        return [x,y]

    intersection = get_intersect(hyperplans[-1], hyperplans[-2])
    plot_grid = {'xmingrid':xmingrid,
                'xmaxgrid':xmaxgrid,
                'ymingrid':ymingrid,
                'ymaxgrid':ymaxgrid,
                'xx':xx,
                'yy':yy,
                'Z':Z,
                'hyperplans':hyperplans,
                'intersection':intersection}

    return plot_grid
def return_annotation_boxes(image, xybox):
    """
    Returns annotation boxes given a classe and a sample
    """
    im = Resize(size=(256, 256))(image)
    im = CenterCrop(size=(224, 224))(im)
    im = OffsetImage(np.array(im), zoom=1)
    annotBox = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    return [annotBox]

def hover_few_shot_space(reduced_text_features, reduced_image_features, images_reshaped, classes, figsize=(15, 15)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    colors = ['red', 'deepskyblue', 'orange', 'green']
    xybox=(25., 25.)

    # Plot Grid
    plot_grid = compute_hyperplans_and_grid(reduced_text_features)
    xx, yy, Z = plot_grid['xx'], plot_grid['yy'], plot_grid['Z']
    plt.contourf(xx.cpu(), yy.cpu(), Z.cpu(), alpha=0.1)

    # Plot hyperplans
    hyperplans, intersection = plot_grid['hyperplans'], plot_grid['intersection']
    ymingrid, ymaxgrid, xmingrid, xmaxgrid = plot_grid['ymingrid'], plot_grid['ymaxgrid'], plot_grid['xmingrid'], plot_grid['xmaxgrid']

    for i, h in enumerate(hyperplans[:]):
        a,b, middle = h
        if a!= np.inf and a!=0:
            if intersection[0]>middle[0]: #intersect with left grid
                plt.plot([intersection[0], xmingrid], [intersection[1], a*xmingrid+b], color=colors[i])
            else:
                plt.plot([intersection[0], xmaxgrid], [intersection[1], a*xmaxgrid+b], color=colors[i])
        if a==np.inf:
            if middle[1].item()<intersection[1]:
                plt.plot([b, b], [ymingrid, intersection[1]], color=colors[i])
            else:
                plt.plot([b, b], [intersection[1], ymaxgrid], color=colors[i])

        if a==0: #b is y
            if middle[0].item()<intersection[0]:
                plt.plot([xmingrid, intersection[0]], [b, b], color=colors[i])
            else:
                plt.plot([intersection[0], xmaxgrid], [b,b], color=colors[i])

    X, Y, colors_list, edgecolor_list, list_image_hover = [], [], [], [], []

    # Support
    for c in range(reduced_text_features.shape[0]):
        support = reduced_text_features[c]
        X.append(support[0].cpu())
        Y.append(support[1].cpu())
        colors_list = colors_list+ [colors[c]]
        edgecolor_list = edgecolor_list+ ['black']
        # Text for support 
        if classes is not None and len(classes)!=0:
            plt.text(support[0].cpu()+0.01, support[1].cpu(), classes[c], fontsize=12)
        
        query = reduced_image_features[c]
        for q in range(query.shape[0]):
            X.append(query[q, 0].cpu())
            Y.append(query[q, 1].cpu())
            colors_list = colors_list+ [colors[c]]
            edgecolor_list = edgecolor_list+ [colors[c]]
            list_image_hover = list_image_hover + return_annotation_boxes(images_reshaped[c][q], xybox)
    X = torch.stack(X)
    Y = torch.stack(Y)

    line = plt.scatter(X, Y, color=colors_list, edgecolor=edgecolor_list)
    plt.contourf(xx.cpu(), yy.cpu(), Z.cpu(), alpha=0.1, color=colors) #cmap=plt.cm.RdBu
    for annotBox in list_image_hover:
        ax.add_artist(annotBox)
        annotBox.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            ind, = line.contains(event)[1]["ind"]
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)

            list_image_hover[ind].xybox = (xybox[0]*ws, xybox[1]*hs)
            list_image_hover[ind].set_visible(True)
            list_image_hover[ind].xy =(X[ind], Y[ind])
            for other_ind in range(len(list_image_hover)):
                if other_ind!=ind:
                    list_image_hover[other_ind].set_visible(False)
        else:
            #if the mouse is not over a scatter point
            for ind in range(len(list_image_hover)):
                list_image_hover[ind].set_visible(False)
        fig.canvas.draw_idle()
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])

    plt.xlim(xmingrid, xmaxgrid)
    plt.ylim(ymingrid, ymaxgrid)
    plt.show()