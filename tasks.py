from main import FCN, ModelRunner


def task2():
    settings = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10
    }

    runner = ModelRunner(settings)
    runner.train()
    # runner.val()
    runner.plot()


def task3_1():
    settings_baseline = {
        'APPLY_TRANSFORMATIONS': False,
        'MODEL': FCN,
        'EPOCHS': 10,
        'LOAD_FROM_PATH': 'baseline_model.model'
    }

    settings_task3_1 = {
        'APPLY_TRANSFORMATIONS': True,
        'MODEL': FCN,
        'EPOCHS': 10
    }

    baseline_runner = ModelRunner(settings_baseline)
    task_model = ModelRunner(settings_task3_1)

    # combine two plots
    task_model.plot(baseline_runner, names=["Baseline", "Model with transformations"])
