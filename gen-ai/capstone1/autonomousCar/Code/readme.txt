Part1:
==========
For Tesla autonomous vehicle detection, It is very difficult to build a self made CNN based model from scratch.
The 1st challenge would be to identify an object that too if it is a vehicle or not itself is bigger challenge.
Then it comes to type of vehicle and no of vehicles.

Given the scenario and objective with annotation csv provided with the assignment.
It has been decided to use an semi/full made ready model and then customize with your own dataset - which looks like the purpose of the assignment.


Therefore, the development direction follows the above idea and among many diff model and technology available,
the YOLOv8 model have been decided to use to train our custom images and dataset.

Here are steps taken:
1: What we have
	a) all the images to get trained
	b) their annotation csv

2: We will use the images but not the csv as csv looks like faulty in my opinion.
	- the reason behind not using existing .csv annotation is:
		- when analyzing, found that there are multiple 'filename' that does not exists as image
                  for example: an image "00000009.jpg" have to have an annotation as "00000009" csv file for the image annotation.
                  but we found many individual "filename" in existing .csv given in assignment for which the image is not available.

		  At this point - it has been decided that - lets create the csv annotation as fresh and then create yollov8 yaml for training data


3: To start with: we have uploaded the images in .rar format with 19MB each to upload in public GitHub repo:
		  [the 19mb size is due to its restriction in upload directly from browser]
	
		A) we have uploaded all images in rar spitted form in GitHub
		B) then read it in code - to download and extract to get all images finally

4: Further, we have started creating
		A) Fresh annotation csv
		B) convert csv to to yaml for final training on yolov8(coco) model
		C) in certain cases, we could individually open the yaml to rectify all its axes using tool like: ''

5: We have used 10 epocs with early drop and saved the model in .pt format to use for future inference

6: Inference ran on saved .pt model - to find near correct prediction with 
		A) Vehicle detection with Box
		B) Identify no of vehicles in road
------------------------------------------------------
The entire code base is made public in GitHub repo: for ease of access and further learning and development


---------
Project DS:

dataset
 |_______ images
            |______ train
	    |_______val
 |________labels
	    |______ train
	    |_______val
 		
multifiles
   |___________ annot
		  |____annotation.csv
   |_____________extractedImg
		   |__________Images
				 |_________*.jpg
		   |__________labels
				|_________ *.txt

		   | _______Images.part1.rar
		   | _______Images.part2.rar
		   | _______Images.part*.rar

=========================================================================================================================	