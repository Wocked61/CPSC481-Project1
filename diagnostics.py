from probability4e import BayesNet, enumeration_ask

T, F = True, False

class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """

    def __init__(self):
        """
        Initialize the Bayesian Network with the nodes and conditional probability
        [cite_start]tables (CPTs) specified in the project description [cite: 6-68].
        """
        self.bn = BayesNet([
            ('Asia', '', 0.01),
            ('Smoking', '', 0.5),
            
            # Tuberculosis depends on Asia
            ('Tuberculosis', 'Asia', {T: 0.05, F: 0.01}),
            
            # Lung Cancer depends on Smoking
            # Using standard Asia network values (0.1/0.01) which fit the diagram structure
            ('LungCancer', 'Smoking', {T: 0.1, F: 0.01}),
            
            # Bronchitis depends on Smoking
            ('Bronchitis', 'Smoking', {T: 0.6, F: 0.3}),
            
            # TBorCancer is a logical OR of Tuberculosis and LungCancer
            ('TBorCancer', 'Tuberculosis LungCancer', {
                (T, T): 1.0, 
                (T, F): 1.0, 
                (F, T): 1.0, 
                (F, F): 0.0
            }),
            
            # Xray depends on TBorCancer
            ('Xray', 'TBorCancer', {T: 0.99, F: 0.05}),
            
            # Dyspnea depends on TBorCancer and Bronchitis
            ('Dyspnea', 'TBorCancer Bronchitis', {
                (T, T): 0.9, 
                (T, F): 0.7, 
                (F, T): 0.8, 
                (F, F): 0.1
            })
        ])

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):
        """
        Compute the most likely disease given the symptoms.
        
        Args:
            visit_to_asia (str): "Yes", "No", or "NA"
            smoking (str): "Yes", "No", or "NA"
            xray_result (str): "Abnormal", "Normal", or "NA"
            dyspnea (str): "Present", "Absent", or "NA"
            
        Returns:
            list: [disease_name (str), probability (float)]
        """
        
        # 1. Prepare Evidence Dictionary
        # Convert string inputs to Boolean values expected by BayesNet
        # "NA" values are omitted from the dictionary (treated as unobserved)
        evidence = {}
        
        if visit_to_asia == "Yes":
            evidence['Asia'] = True
        elif visit_to_asia == "No":
            evidence['Asia'] = False
            
        if smoking == "Yes":
            evidence['Smoking'] = True
        elif smoking == "No":
            evidence['Smoking'] = False
            
        if xray_result == "Abnormal":
            evidence['Xray'] = True
        elif xray_result == "Normal":
            evidence['Xray'] = False
            
        if dyspnea == "Present":
            evidence['Dyspnea'] = True
        elif dyspnea == "Absent":
            evidence['Dyspnea'] = False

        # 2. Query the Network for each disease
        # We need to find P(Disease=True | Evidence) for all three diseases
        target_diseases = {
            'Tuberculosis': 'TB',        # Map internal node name to output string
            'LungCancer': 'Cancer',
            'Bronchitis': 'Bronchitis'
        }
        
        best_disease = None
        best_prob = -1.0

        for node_name, output_name in target_diseases.items():
            # enumeration_ask returns a probability distribution object
            dist = enumeration_ask(node_name, evidence, self.bn)
            
            # We want the probability that the disease is Present (True)
            prob = dist[True]
            
            # Track the highest probability
            if prob > best_prob:
                best_prob = prob
                best_disease = output_name

        # 3. Return the result in the specified format
        return [best_disease, best_prob]