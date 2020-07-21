def ClearBlackEdge(image):
    x = image.shape[1]
    y = image.shape[0]
    edges_x=[]
    edges_y=[]
    edges_x_up=[]
    edges_y_up=[]
    edges_x_down=[]
    edges_y_down=[]
    edges_x_left=[]
    edges_y_left=[]
    edges_x_right=[]
    edges_y_right=[]

    for i in range(x):
        for j in range(y):
            if int(image[j][i])>250:
                edges_x_left.append(i)
                edges_y_left.append(j)
        if len(edges_x_left)!=0 or len(edges_y_left)!=0:
            break

    for i in range(x):
        for j in range(y):
            if int(image[j][x-i-1])>250:
                edges_x_right.append(i)
                edges_y_right.append(j)
        if len(edges_x_right)!=0 or len(edges_y_right)!=0:
            break
        
    for j in range(y):
        for i in range(x):
            if int(image[j][i])>250:
                edges_x_up.append(i)
                edges_y_up.append(j)
        if len(edges_x_up)!=0 or len(edges_y_up)!=0:
            break

    for j in range(y):
        for i in range(x):
            if int(image[y-j-1][i])>250:
                edges_x_down.append(i)
                edges_y_down.append(j)
        if len(edges_x_down)!=0 or len(edges_y_down)!=0:
            break
        
    edges_x.extend(edges_x_left)
    edges_x.extend(edges_x_right)
    edges_x.extend(edges_x_up)
    edges_x.extend(edges_x_down)
    edges_y.extend(edges_y_left)
    edges_y.extend(edges_y_right)
    edges_y.extend(edges_y_up)
    edges_y.extend(edges_y_down)

    left=min(edges_x)               #左边界
    right=max(edges_x)              #右边界
    bottom=min(edges_y)             #底部
    top=max(edges_y)                #顶部

    image2=image[bottom:top,left:right]
    return image2


# img = cv2.imread('18-36-10.jpg',0)
# noEdge = remove_black_edges_optimization(img)


# plt.figure()
# plt.imshow(img,'image')
# plt.figure()
# plt.imshow(noEdge,'image')
# plt.show()