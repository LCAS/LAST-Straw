# Supplementary code associated with the LAST-Straw Dataset for the creation and assessment of skeletons

## Steps to reproduce 

### Production of skeletons:
- Download data from the [website](https://lcas.github.io/LAST-Straw/)
- Use data_loader to filter the annotated samples and extract petiole instances
- Generate results from Xu et al (shortest path) and ground truth (smoothed/adjusted/placed manually)
    - use [PlantScan3D](https://plantscan3d.readthedocs.io/en/latest/index.html) 
    - parse skeletons to ply using parse_plantscan3d
- Generate results for SOM: 
    - clone [4d_plant_registration*](https://github.com/KatherineJames/4d_plant_registration) into project 
    - run strawberry_skel in
- Generate results for L1:
    - clone [L1-Skeleton**](https://github.com/KatherineJames/L1-Skeleton) into project and install dependencies
    - run ./generate_l1_results.sh

*forked from https://github.com/PRBonn/4d_plant_registration
**forked from https://github.com/jasonkena/L1-Skeleton.git

### Trait measurement:
- measure.py - measure the length of individual petioles (out: Results/{sample}_length.txt) 
- assess.py*: generate dense graphs and compute quantitative metrics (out: Results/{method}_assessment.txt)
- re_id_stems.py: associate petioles with leaves
- track.py: plot graphs for stem length over time

*Clone GraphMatching3D repo into the project (https://github.com/LCAS/GraphMatching3D)

### Other:
- view_results.py: view a particular skeleton plotted in Matplotlib
- view_plant_skel.py: view selected skeletons superimposed with colour