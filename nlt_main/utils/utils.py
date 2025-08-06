from psychopy import visual, core, event, gui
import pandas as pd
import random
from datetime import datetime
from dateutil import parser

def collect_participant_info():
    """
    Collects participant information through a dialog box.
    """
    dlg = gui.Dlg(title="Participant Information")
    dlg.addText('Please enter the following info:')
    dlg.addField('Participant ID:')
    dlg.addField('생년월일 (YYYY-MM-DD):')
    dlg.addField('성별:', choices=['남성', '여성'])

    ok_data = dlg.show()
    
    if dlg.OK:
        info = {
            'Participant ID': ok_data[0],
            'Birthdate': parser.parse(ok_data[1], yearfirst=True, dayfirst=False).date(),
            'Gender': ok_data[2],
        }
        return info
    else:
        core.quit()

def save_results(dir, results, info, filename='results.csv'):
    """
    Save the experiment results to a CSV file.
    Each result will be appended with participant information.
    """
    for result in results:
        result.update(info)
    df = pd.DataFrame(results)
    df.to_csv(f'{dir}/{filename}', index=False)