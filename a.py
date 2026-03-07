from roboflow import Roboflow

rf = Roboflow(api_key="KngbnnKbZUYoveXna0Wd")
workspace = rf.workspace("vinay-kasana")

workspace.deploy_model(
  model_type="best.pt",
  model_path="C:\\Users\\Vinay Kasana\\OneDrive\\Desktop\\minor\\Autonomous-Parking-System\\best.pt",
  project_ids=["project-1", "project-2", "project-3"],
  model_name="my-custom-model"
)