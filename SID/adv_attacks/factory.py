import sys
sys.path.append("../")

from AArate.autoattack import AutoAttack


def create_attack(model, adv_method, adv_params, **kwargs):
    for k in ['eps','alpha','step_size']:
        if k in adv_params.keys():
            adv_params[k] = eval(adv_params[k])

    if adv_method in ['FAB']:
        assert kwargs.get('num_classes',False), 'FAB needs to define the number of classes.'
        adv_params['n_classes'] = kwargs['num_classes']

    if adv_method == "AutoAttack":
        atk = AutoAttack(model, **adv_params)
    else:
        atk = __import__('torchattacks').__dict__[adv_method](model=model, **adv_params)

    return atk