#rsync --progress -auzhe 'ssh' * aipg-demo02:/home/bduser/topologies/3D_UNet/keras_training_only_version/ 
#rsync --progress -auzhe 'ssh' * aipg-demo03:/home/bduser/topologies/3D_UNet/keras_training_only_version/
#rsync --progress -auzhe 'ssh' * aipg-demo04:/home/bduser/topologies/3D_UNet/keras_training_only_version/

ssh demo02 "cd topologies && git pull"

ssh demo03 "cd topologies && git pull"

ssh demo04 "cd topologies && git pull"


