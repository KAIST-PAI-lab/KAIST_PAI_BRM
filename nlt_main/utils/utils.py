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
    dlg.addField('Name:')
    dlg.addField('Birthdate (YYYY-MM-DD):')
    dlg.addField('Gender:', choices=['Male', 'Female', 'Other'])
    dlg.addField('Handedness:', choices=['Right', 'Left', 'Ambidextrous'])
    dlg.addField('Major/Grade:')

    ok_data = dlg.show()
    
    if dlg.OK:
        info = {
            'Participant ID': ok_data[0],
            'Name': ok_data[1],
            'Birthdate': parser.parse(ok_data[2], yearfirst=True, dayfirst=False).date(),
            'Gender': ok_data[3],
            'Handedness': ok_data[4],
            'Major/Grade': ok_data[5]
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