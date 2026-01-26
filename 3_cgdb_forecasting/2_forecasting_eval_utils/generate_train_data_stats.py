from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction
from digital_twin_converter.common.config import Config
import argparse
import wandb


DEBUG = False


all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]



def main(base_save_path: str):

    wandb.init(project="genie-dt-cgdb-baselines-forecasting", group="Data Generation")
    wandb.run.name = "Data Stats Generation"

    for indication in all_indications:

        print(f"Generating train data stats for {indication}")

        save_path_for_variable_stats = base_save_path + f"{indication}_train_data_stats.csv"

        config = Config()

        dm = SingleIndicationDataManager(indication, config=config)
        dm.load_indication_data()
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_dataset_splits()

        data_splitter = DataSplitterForecasting(data_manager=dm, config=config,
                                                save_path_for_variable_stats=save_path_for_variable_stats)
        data_splitter.setup_statistics()

        print(f"Saved train data stats for {indication} to {save_path_for_variable_stats}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate train data stats for all indications")
    parser.add_argument("--base_save_path", type=str, default="cgdb_baselines/forecasting/train_data_stats/",
                        help="Base path to save the train data stats")
    args = parser.parse_args()

    main(args.base_save_path)


