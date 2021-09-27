import vibe
import updated_flow
import cv2
import numpy as np

FRAMES = None
eval_start = None
eval_end = None

kernel = np.ones((3,3),np.uint8)

def remove_shadows(image):
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 35)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def pre_process(frame):
    fr = cv2.medianBlur(frame, 3)
    fr = unsharp_mask(fr)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

    return fr

def post_process(frame, th = 10):
    # fr = cv2.medianBlur(frame, 3)
    # ret, fr = cv2.threshold(frame, th, 255, cv2.THRESH_BINARY)
    # fr = cv2.dilate(fr, np.ones((3,3),np.uint8))
    fr = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    fr = cv2.morphologyEx(fr, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

    return fr

def post_process2(frame, th = 10):
    # fr = cv2.medianBlur(frame, 3)
    ret, fr = cv2.threshold(frame, th, 255, cv2.THRESH_BINARY)
    fr = cv2.dilate(fr, kernel)
    # fr = cv2.morphologyEx(fr, cv2.MORPH_OPEN, kernel)
    # fr = cv2.morphologyEx(fr, cv2.MORPH_CLOSE, kernel)

    return fr

def static_bg(args,R1,rand_samples_1,boot_strap_1,n_min_1):
    bs = vibe.Vibe_Subsense(R=R1, rand_samples=rand_samples_1, bootstrap=boot_strap_1, n_min=n_min_1)
    bs2 = cv2.createBackgroundSubtractorMOG2()
    bs3 = cv2.createBackgroundSubtractorKNN()

    of = updated_flow.OpticalFlow()
    bg = updated_flow.ModelBackground()

    frame_name = "in" + str(1).zfill(6) + ".jpg"
    frame1 = cv2.imread(args.inp_path + frame_name)
    #frame1 = cv2.resize(frame1, (320,240))
    prvs = pre_process(frame1)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    for i in range(1, FRAMES+1):
        # print(i)
        frame_name = "in" + str(i).zfill(6) + ".jpg"
        frame = cv2.imread(args.inp_path + frame_name)
        #frame = cv2.resize(frame, (320,240))
        # frame = remove_shadows(frame)
        
        frame = cv2.GaussianBlur(frame,(3,3),0)
        mask_mog = bs2.apply(frame)
        mask_knn = bs3.apply(frame)
        mask = post_process(mask_knn)

        of.insert_image(frame)
        flow_magnitude = of.compute_optical_flow()
        mask_of = of.get_mask_of_moving()
        bg.insert_image(frame)
        bg_img = bg.get_background_image()
        mask_bg = bg.get_mask_of_foreground(frame)
        mask2 = (mask_bg.reshape(mask_bg.shape[0],-1)* mask_of)**(0.9)
        mask_fin = updated_flow.mask2gray(mask2)
        ret, mask_fin = cv2.threshold(mask_fin, 7, 255, cv2.THRESH_BINARY)
        
        # if res is not None:
        #     res = cv2.medianBlur(res, 5)
        #     res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        #     res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

        pred_name = "gt" + str(i).zfill(6) + ".png"
        if i >= eval_start and i <= eval_end:
            cv2.imwrite(args.out_path + pred_name, mask & mask_fin)


def background_perform(args,R1,rand_samples_1,boot_strap_1,n_min_1):
        
    bs = vibe.Vibe_Subsense(R=R1, rand_samples=rand_samples_1, bootstrap=boot_strap_1, n_min=n_min_1)
    
    frame_name = "in" + str(1).zfill(6) + ".jpg"
    frame1 = cv2.imread(args.inp_path + frame_name)
    # frame1 = cv2.resize(frame1, (320,240))
    prvs = pre_process(frame1)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    of = updated_flow.OpticalFlow()
    bg = updated_flow.ModelBackground()
    backSub_mog = cv2.createBackgroundSubtractorMOG2(history=40,detectShadows=False)
    backSub_knn = cv2.createBackgroundSubtractorKNN(history=40,detectShadows=False)


    for i in range(1, FRAMES+1):
        frame_name = "in" + str(i).zfill(6) + ".jpg"
        frame = cv2.imread(args.inp_path + frame_name)
        # frame = cv2.resize(frame, (320,240))
        frame2 = frame
        
        frame = remove_shadows(frame)
        frame = pre_process(frame)

        # cv2.imshow("frame", frame)
        # keyboard = cv2.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     break

        # flow = cv2.calcOpticalFlowFarneback(prvs, frame, None, 0.5, 10, 20, 3, 7, 1.5, 0)
        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # bgr = post_process(bgr)
        
        mog = backSub_mog.apply(frame)
        knn = backSub_knn.apply(frame)

        of.insert_image(frame2)
        flow_magnitude = of.compute_optical_flow()
        mask_of = of.get_mask_of_moving()
        bg.insert_image(frame2)
        bg_img = bg.get_background_image()
        mask_bg = bg.get_mask_of_foreground(frame2)
        mask2 = (mask_bg.reshape(mask_bg.shape[0],-1)* mask_of)**(0.9)
        mask_fin = updated_flow.mask2gray(mask2)
        ret, store_fin = cv2.threshold(mask_fin, 25, 255, cv2.THRESH_BINARY)
        ret, mask_fin = cv2.threshold(mask_fin, 45, 255, cv2.THRESH_BINARY)
        tmp = post_process(mog,10) & post_process(knn,10) & mask_fin
        # bs.apply(frame, tmp) 


        # res = bs.get('get_image')

            
        # if not res is None:
        #     mask_fin = mask_fin & res
        #     mask_fin = cv2.medianBlur(mask_fin, 5)  
        #     # mask_fin = post_process(mog,10) & post_process(knn,10) & mask_fin
        #     # mask_fin = cv2.dilate(mask_fin, (3,3))
        #     # mask_fin = cv2.morphologyEx(mask_fin, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        #     mask_fin = cv2.resize(mask_fin, (320,240))
        #     pred_name = "gt" + str(i).zfill(6) + ".png"
        #     if i >= eval_start and i <= eval_end:
        #         cv2.imwrite(args.out_path + pred_name, mask_fin)

        pred_name = "gt" + str(i).zfill(6) + ".png"
        if i >= eval_start and i <= eval_end:
            ans = cv2.resize((post_process2(mog,10) | post_process2(knn,10)) & mask_fin, (320,240))
            cv2.imwrite(args.out_path + pred_name, ans)
                

def background_perform2(args,R1,rand_samples_1,boot_strap_1,n_min_1):
        
    bs = vibe.Vibe_Subsense(R=R1, rand_samples=rand_samples_1, bootstrap=boot_strap_1, n_min=n_min_1)
    bs2 = cv2.createBackgroundSubtractorMOG2()
    bs3 = cv2.createBackgroundSubtractorKNN()
    
    frame_name = "in" + str(1).zfill(6) + ".jpg"
    frame1 = cv2.imread(args.inp_path + frame_name)
    #frame1 = cv2.resize(frame1, (320,240))
    prvs = pre_process(frame1)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    for i in range(1, FRAMES+1):
        # print(i)
        frame_name = "in" + str(i).zfill(6) + ".jpg"
        frame = cv2.imread(args.inp_path + frame_name)
        #frame = cv2.resize(frame, (320,240))
        frame2 = frame
        mask_mog = bs2.apply(frame)
        mask_knn = bs3.apply(frame)
        mask = mask_knn | mask_mog
        
        frame = pre_process(frame)
        # flow = cv2.calcOpticalFlowFarneback(prvs, frame, None, 0.5, 10, 20, 3, 7, 1.5, 0)
        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # bgr = post_process(bgr)        
        bs.apply(frame,mask) 
        res = bs.get('get_image')
            
        if not res is None:
            res = res & mask
            res = cv2.medianBlur(res, 5)
            res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            res = cv2.dilate(res, np.ones((3,3),np.uint8))
            pred_name = "gt" + str(i).zfill(6) + ".png"
            if i >= eval_start and i <= eval_end:
                cv2.imwrite(args.out_path + pred_name, res)
           

def ptz_bg_model(args,R1,rand_samples_1,boot_strap_1,n_min_1):
    bs = vibe.Vibe_Subsense(R=R1, rand_samples=rand_samples_1, bootstrap=boot_strap_1, n_min=n_min_1)
    
    frame_name = "in" + str(1).zfill(6) + ".jpg"
    frame1 = cv2.imread(args.inp_path + frame_name)
    # frame1 = cv2.resize(frame1, (320,240))
    prvs = pre_process(frame1)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    of = updated_flow.OpticalFlow()
    bg = updated_flow.ModelBackground()
    backSub_mog = cv2.createBackgroundSubtractorMOG2()
    backSub_knn = cv2.createBackgroundSubtractorKNN()


    for i in range(1, FRAMES+1):
        # print(i)
        frame_name = "in" + str(i).zfill(6) + ".jpg"
        frame = cv2.imread(args.inp_path + frame_name)
        # frame = cv2.resize(frame, (320,240))
        frame2 = frame
        
        frame = remove_shadows(frame)
        frame = pre_process(frame)

        # cv2.imshow("frame", frame)
        # keyboard = cv2.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     break

        # flow = cv2.calcOpticalFlowFarneback(prvs, frame, None, 0.5, 10, 20, 3, 7, 1.5, 0)
        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # bgr = post_process(bgr)
        
        mog = backSub_mog.apply(frame)
        knn = backSub_knn.apply(frame)

        of.insert_image(frame2)
        flow_magnitude = of.compute_optical_flow()
        mask_of = of.get_mask_of_moving()
        bg.insert_image(frame2)
        bg_img = bg.get_background_image()
        mask_bg = bg.get_mask_of_foreground(frame2)
        mask2 = (mask_bg.reshape(mask_bg.shape[0],-1)* mask_of)**(0.9)
        mask_fin = updated_flow.mask2gray(mask2)
        ret, store_fin = cv2.threshold(mask_fin, 25, 255, cv2.THRESH_BINARY)
        ret, mask_fin = cv2.threshold(mask_fin, 45, 255, cv2.THRESH_BINARY)
        tmp = post_process(mog,10) & post_process(knn,10) & mask_fin
        bs.apply(frame, tmp) 


        res = bs.get('get_image')

            
        if not res is None:
            mask_fin = mask_fin & res
            mask_fin = cv2.medianBlur(mask_fin, 5)  
            # mask_fin = post_process(mog,10) & post_process(knn,10) & mask_fin
            # mask_fin = cv2.dilate(mask_fin, (3,3))
            # mask_fin = cv2.morphologyEx(mask_fin, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            mask_fin = cv2.resize(mask_fin, (320,240))
            pred_name = "gt" + str(i).zfill(6) + ".png"
            if i >= eval_start and i <= eval_end:
                cv2.imwrite(args.out_path + pred_name, mask_fin)

        # pred_name = "gt" + str(i).zfill(6) + ".png"
        # if i >= eval_start and i <= eval_end:
        #     ans = cv2.resize((post_process(mog,10) | post_process(knn,10)) & mask_fin, (320,240))
        #     cv2.imwrite(args.out_path + pred_name, ans)
    
