import numpy as np
# 密度图合并，这个思路就先列合并再行合并
def merge(out,num,padding):
    row = num
    col = num
    row_out = []
    guodu = 0
    length = (out[0].shape)[0]-padding

    top_row = int((out[0].shape)[0]-padding-guodu)
    left_col = int((out[0].shape)[1]-padding-guodu)

    for i in range(row):
        
        row_out.append(out[i*col][0:top_row,:])
        for j in range(col-1):
            mid1_up = int(length-guodu) if j == 0 else int(length + padding - guodu)
            mid1_down = int(length+guodu) if j == 0 else int(length + padding + guodu)
            mid2_up = int(padding - guodu)
            mid2_down = int(padding + guodu)

            row_mid1 = out[i*col + j][mid1_up:mid1_down,:]
            row_mid2 = out[i*col + j+1][mid2_up:mid2_down,:]
            row_mid = row_mid1 + row_mid2 
        
            down_up = int(padding + guodu)
            down_down = int(length+padding - guodu) if j < col-2 else int(length+padding)
      
            row_down = out[i*col + j+1][down_up:down_down,:]
            row_out[i] = np.concatenate((row_out[i],row_mid) ,axis = 0)
            row_out[i]= np.concatenate((row_out[i],row_down) ,axis = 0)

        if i == 0 :
            #all_out.append(row_out[0])
            all_out = row_out[0][:,0:left_col]
    
        if i > 0:
            mid1_left = int(length-guodu) if i == 1 else int(length + padding - guodu)
            mid1_right = int(length+guodu) if i == 1 else int(length + padding + guodu)
            mid2_left = int(padding - guodu)
            mid2_right = int(padding+guodu)

            col_mid1 = row_out[i-1][:,mid1_left:mid1_right]
            col_mid2 = row_out[i][:,mid2_left:mid2_right]
            col_mid = 0.2*col_mid1 + 0.8*col_mid2 

            right_left = int(padding + guodu)
            right_right = int(length+padding-guodu) if i < row-1 else int(length+padding)
            col_right = row_out[i][:,right_left:right_right]

    
            all_out = np.concatenate((all_out,col_mid) , axis = 1)
            all_out = np.concatenate((all_out,col_right) , axis = 1)
    
    

    #print(all_out)
    return all_out