from run_end2end_evaluation import End2EndEvaluator


def main():
	config_path = "./cfg/end2end_config.json"
	evaluator = End2EndEvaluator(config_path)
	raw_model = evaluator.raw_model
	supine_model = evaluator.supine_model
	lying_left_model = evaluator.lying_left_model
	lying_right_model = evaluator.lying_right_model
	print(supine_model)

if __name__ == "__main__":
	main()