### WebUI Instructions
#### 1. Start WebUI
Here, we take the face scene as an example. The file structure is as follows:
```
└── data
    └── face
        ├── image
        ├── sparse
        └── point_cloud.ply
```
Use the command to start the WebUI
```shell
python webui.py --colmap_dir ./data/face/ --gs_source ./data/face/point_cloud.ply --output_dir result
```
Afterward, you will see the following screen. \
<img width="600" src="./webui_init.png">

#### 2. Select the Edit Area
Check the boxes for `Enable Filter` and `Draw Filter Box` on the right side, then click on the screen to select the area. \
<img width="600" src="./webui_select_mask1.png">

Then, press `Filter Now!` to confirm the selected area, which will be highlighted in red. \
<img width="600" src="./webui_select_mask2.png">

You can also switch to other viewpoints and continue selecting the edit area; we will take the intersection of all selected areas. Additionally, you can reset the edit area by clicking `Reset Filter!`. \
<img width="600" src="./webui_select_mask3.png">

#### 3. Select Drag Control Points and Target Points
Check the boxes for `Drag Point Enabled` and `Add Point` on the right side, then click on the screen to select points. Here, we use the rendered depth map for re-projection to obtain 3D points. \
<img width="600" src="./webui_select_point1.png">

Next, press `Add Drag Point` To Text to add the selected points to the text box. You can fine-tune the positions of the selected points in the text box, or continue selecting points on the screen and add them one by one to the text box. \
<img width="200" src="./webui_select_point2.png">

Finally, press `Input Drag Point` to use the content in the text box as input. All input point pairs will be displayed on the screen with blue (control points) and red (target points). \
<img width="600" src="./webui_select_point3.png">

#### 4. Training

After selecting the edit area and drag points, press `Drag Now!` to start training. \
Alternatively, you can press Save Mask&Point to export the data, with the export path being `output_dir/export/`. The exported Mask and Points will be `gaussian_mask.pt` and `drag_points.json`, respectively. \
You can then train without using the WebUI by using the following command.

```shell
python drag_3d.py --config configs/main.yaml \
                  --colmap_dir ./data/face/ \
                  --gs_source ./data/face/point_cloud.ply \
                  --point_dir ./data/face/export_1/drag_points.json \
                  --mask_dir ./data/face/export_1/gaussian_mask.pt \
                  --output_dir result
```
