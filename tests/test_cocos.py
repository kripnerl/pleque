import pleque.core.cocos as coc

def test_cocos_dict():
    for i in range(1, 9):
        for cocos_idx in [i, 10+i]:
            cocos = coc.cocos_coefs(cocos_idx)

            for k in ['exp_Bp', 'sigma_Bp', 'sigma_cyl', 'sigma_pol', 'sign_q', 'sign_pprime']:
                assert k in cocos

            if cocos_idx >= 10:
                assert cocos["exp_Bp"] == 1
            else:
                assert cocos["exp_Bp"] == 0


            if cocos_idx in [1, 11]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [2, 12]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [3, 13]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == +1

            if cocos_idx in [4, 14]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == +1

            if cocos_idx in [5, 15]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [6, 16]:
                assert cocos['sigma_Bp'] == +1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == -1
                assert cocos['sign_q'] == -1
                assert cocos['sign_pprime'] == -1

            if cocos_idx in [7, 17]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == +1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == +1

            if cocos_idx in [8, 18]:
                assert cocos['sigma_Bp'] == -1
                assert cocos['sigma_cyl'] == -1
                assert cocos['sigma_pol'] == +1
                assert cocos['sign_q'] == +1
                assert cocos['sign_pprime'] == +1