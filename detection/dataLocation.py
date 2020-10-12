roots= {
        "icdar":{
            "root":"/content/icda_table_general_dataset_img",
            "image":"Images",
            "trainjson":["train_icdar_tables"],
            "testjson":["test_icdar_table"]
          },
        "bank":{
            "root":"/content/VOC2007",
            "image":"JPEGImages",
            "trainjson":["HDFC_train_coco","ICICI_train_coco"],
            "testjson":["HDFC_test_coco","ICICI_test_coco"]

        }
       }

def getRoots():
  return roots