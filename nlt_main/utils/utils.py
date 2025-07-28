from psychopy import visual, core, event, gui
import pandas as pd
import random
from datetime import datetime

def collect_participant_info():
    """
    Collects participant information through a dialog box.
    """
    dlg = gui.Dlg(title="Participant Information")
    dlg.addText('Please enter the following info:')
    dlg.addField('Name:')
    dlg.addField('Participant ID:')
    dlg.addField('Birthdate (YYYY-MM-DD):')
    dlg.addField('Gender:', choices=['Male', 'Female', 'Other'])
    dlg.addField('Handedness:', choices=['Right', 'Left', 'Ambidextrous'])
    dlg.addField('Major/Grade:')

    ok_data = dlg.show()
    if dlg.OK:
        info = {
            'Name': ok_data[0],
            'Participant ID': ok_data[1],
            'Birthdate': datetime.strptime(ok_data[2], '%Y-%m-%d'),
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