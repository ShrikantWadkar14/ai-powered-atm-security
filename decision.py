from types import SimpleNamespace

class DecisionEngine:
    def __init__(self):
        self.threshold_high = 2.0

    def evaluate(self, persons, weapons, tamper_res, action_res):
        score = 0.0
        reasons = []
        if len(persons) > 1:
            score += 1.0; reasons.append('multiple_persons')
        if weapons and len(weapons) > 0:
            score += 2.0; reasons.append('weapon_detected')
        if tamper_res.get('covered'):
            score += 2.0; reasons.append('camera_tamper')
        if action_res.get('loitering'):
            score += 0.8; reasons.append('loitering')
        for a in action_res.get('actions', []):
            if a.get('type') in ('violent_motion','possible_faint'):
                score += 1.5; reasons.append(a.get('type'))
        if score >= self.threshold_high:
            return SimpleNamespace(raise_alert=True, level='HIGH', reasons=reasons)
        elif score > 0:
            return SimpleNamespace(raise_alert=True, level='SUSPICIOUS', reasons=reasons)
        else:
            return SimpleNamespace(raise_alert=False, level='NORMAL', reasons=[])
