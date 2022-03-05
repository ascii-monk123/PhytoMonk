def quantify(leaf, disease_mask):
    totalPix = 0
    disease_pix = 0

    for pixel in leaf.reshape(-1):
        if pixel == 255:
            totalPix+=1
    
    for pixel in disease_mask.reshape(-1):
        if pixel == 255:
            disease_pix+=1

    return (disease_pix / totalPix) * 100

