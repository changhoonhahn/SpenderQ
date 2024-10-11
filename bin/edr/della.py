'''

python script for deploying job on adroit 

'''
import os, sys
import time 


def train_spenderq(study, tag, zmin, zmax, debug=False):
    ''' deploy SpenderQ training 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % study,
        ["#SBATCH --time=11:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH -o o/%s.o" % study, 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --constraint=gpu80", 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "",
        "python /home/chhahn/projects/SpenderQ/bin/train_spender.py /tigress/chhahn/spender_qso/train /tigress/chhahn/spender_qso/models/%s.pt -t %s -n 10 -zmin %f -zmax %f -l 100 -v" % (study, tag, zmin, zmax),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


def postprocess(model_name, input_tag, output_tag, ibatch0, ibatch1, sigma=1.5, debug=True): 
    ''' postprocess SpenderQ outputs 
    '''
    # data direcotry
    dir_dat = '/tigress/chhahn/spender_qso/train'

    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_%s.%i_%i" % (input_tag, output_tag, ibatch0, ibatch1),
        "#SBATCH -o o/%s_%s.%i_%i.o" % (input_tag, output_tag, ibatch0, ibatch1), 
        ["#SBATCH --time=05:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --constraint=gpu80", 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate gqp", 
        "",
        "postprocess=/home/chhahn/projects/SpenderQ/bin/postprocess.py", 
        "for ibatch in {%i..%i}; do python $postprocess %s /tigress/chhahn/spender_qso/models/%s.pt -ti %s -to %s -i $ibatch -sigma %f; done" % (ibatch0, ibatch1, dir_dat, model_name, input_tag, output_tag, sigma),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_spender.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _spender.slurm')
    os.system('rm _spender.slurm')
    return None 


if __name__=="__main__": 
    # edr 
    #postprocess('qso.edr.z_2p1_3p5', # model name
    #        'DESIedr.qso_highz', 
    #        'edr.highz.iter0', 
    #        0, 26, sigma=1.5, debug=True) 
    
    #train_spenderq(
    #        'qso.london.z_2p1_3p5.rebin.iter4', 
    #        'london_highz.rebin.iter4', 
    #        2.1, 3.5, debug=False)
