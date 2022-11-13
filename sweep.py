
sweep_configuration = {
    'method': 'grid',
    'name': 'NMI weight Sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'Epoch Loss'
		},
    'parameters': {
        'weight': {'values': [0.01, 0.05, 0.1, 0.5]}
     }
}