# For basic functionality
import pystan, numpy, patsy, os

# For print functionality
import re, io, contextlib

class EdstanData:
    
    def __init__(self, response_matrix=None, item_id=None, person_id=None, 
                 y=None, person_data=None, formula=None):
        
        # Convert basic parts to numpy arrays and get I and J
        if response_matrix is None:
            y_raw = numpy.asarray(y)
            ii_raw = numpy.asarray(item_id)
            jj_raw = numpy.asarray(person_id)
            I = max(ii_raw)
            J = max(jj_raw)
        else:
            response_matrix = numpy.asarray(response_matrix)
            I = response_matrix.shape[1]
            J = response_matrix.shape[0]
            y_raw = response_matrix.flatten()
            ii_raw = numpy.tile(numpy.arange(1, I+1), J)
            jj_raw = numpy.repeat(numpy.arange(1, J+1), I)
        
        #!! Check length of y, ii, jj
        #!! Check each start at 0 or 1
        #!! Check each are consecutive integers
        
        if person_data is None and formula is not None:
            # Gave formula without person_data
            raise NameError('person_matrix is required with the formula argument')
        elif person_data is None and formula is None:
            # Gave neither formula nor person_data
            W_raw = numpy.broadcast_to(1, (J, 1))
        elif person_data is not None and formula is None:
            # Gave person_data without formula
            try: W_raw = numpy.array(person_data)
            except: raise NameError('person_data must be convertable to a numpy matrix when formula is not provided')            
            if W_raw.ndim == 1: W_raw =  W_raw.reshape((len(W_raw),1))
            elif W_raw.ndim == 2: pass
            else: raise NameError('person_data must be be a matrix (two-dimensional array)')
            if W_raw.shape[0] != J and W_raw.shape[0] != len(jj_raw):
                raise NameError('person_data must have number of rows equal to the number of persons or number of elements in person_id')
        else:
            # Gave person_data and formula
            try: W_raw = patsy.dmatrix(formula, person_data, return_type='matrix')
            except: raise NameError('Could not apply patsy formula to person_data')
            W_raw = numpy.array(W_raw)

        # Reduce person covariate matrix to one row per person if needed
        K = W_raw.shape[1]
        if W_raw.shape[0] == J:
            W = W_raw
        else:
            first_j = [None]*J
            for idx, j in enumerate(jj_raw) :
                if not numpy.isnan(j) and first_j[j-1] is None:
                    first_j[j-1] = idx
            W = W_raw[first_j, ]
        
        # Ensure model intercept is included
        if numpy.all(W[:,0] == 1) == False:
            raise NameError('The design matrix for the latent regression must have a first column with all elements equal to 1')
        
        # Remove missing data
        not_missing = numpy.isnan(y_raw) == False
        N = sum(not_missing)
        ii = ii_raw[not_missing]
        jj = jj_raw[not_missing]
        y = y_raw[not_missing]

        self.datalist = {'N': N, 'I': I, 'J': J, 'y': y, 'ii': ii, 'jj': jj, 
                         'K': K, 'W': W}
        self.__table__ = 'No model fit'
    
    def __str__(self):
        
        #!! Do something with this?
       pass
    
    def max_score_item(self):
        
        max_score = [0] * self.datalist['I']
        for n in range(self.datalist['N']):
            i = self.datalist['ii'][n] - 1
            max_score[i] = max(self.datalist['y'][n], max_score[i])
        return max_score

    def beta_key(self, rating_scale=False):
        
        if rating_scale is True:
            n_betas_per_item = numpy.tile(1, self.datalist['I'])
        else:
            n_betas_per_item = self.max_score_item()
        key = []
        start = 0
        for m in n_betas_per_item:
            end = start + m 
            key.append(range(start, end))
            start = end
        return key
            
    def raw_scores(self):
        
        raw_scores = [0] * self.datalist['J']
        for n in range(self.datalist['N']):
            j = self.datalist['jj'][n] - 1
            raw_scores[j] = raw_scores[j] + self.datalist['y'][n]
        return raw_scores
    
    def fit_model(self, model, output=True, **kwargs):
        
        def full_path(file):
            here = os.path.dirname(__file__)
            return os.path.join(here, 'models', file)
        
        if model.endswith('.stan'):
            file = model
            model_name = ''
        elif model == 'rasch':
            file = full_path('rasch_latent_reg.stan')
            model_name = 'Rasch model'
        elif model == 'rsm':
            file = full_path('rsm_latent_reg.stan')
            model_name = 'Rating scale model'
        elif model == 'pcm':
            file = full_path('pcm_latent_reg.stan')
            model_name = 'Partial credit model'
        elif model == '2pl':
            file = full_path('2pl_latent_reg.stan')
            model_name = 'Two-parameter logistic model'
        elif model == 'grsm':
            file = full_path('grsm_latent_reg.stan')
            model_name = 'Generalized rating scale model'
        elif model == 'gpcm':
            file = full_path('gpcm_latent_reg.stan')
            model_name = 'Generalized partial credit model'
        else:
            raise NameError('Unrecognized model')   
        
        fit = pystan.stan(file=file, data=self.datalist, **kwargs)
        
        if output: 
            print(model_name)
            self.print_from_fit(fit)
        
        return fit
    
    def print_from_fit(self, fit):
        
        raw_print = io.StringIO()
        with contextlib.redirect_stdout(raw_print):
            print(fit)
        lines = re.split('\n', raw_print.getvalue())
        
        def grab_index(parameter, lines, end = '[\s\[].*$'):
            search_string = '^' + parameter + end
            index = [i for i,line in enumerate(lines) if re.search(search_string, line)]
            return index
        
        def print_lines(index, lines, pad = ''):
            if isinstance(index, list) == False:
                index = [index]
            for i in index:
                print(pad + lines[i])
        
        index_blanks = grab_index('', lines, '$')
        index_top = list(range(0, index_blanks[0]+1))
        index_bottom = list(range(index_blanks[1], len(lines)-1))
        index_header = grab_index('\s+mean', lines)
        index_alpha = grab_index('alpha', lines)
        index_beta = grab_index('beta', lines)
        index_kappa = grab_index('kappa', lines)
        index_lambda = grab_index('lambda', lines)
        index_sigma = grab_index('sigma', lines)
        
        beta_key = self.beta_key(rating_scale = len(index_kappa) > 0)
        
        print_lines(index_top, lines)
        print_lines(index_header, lines, '  ')    
        for i,item in enumerate(beta_key):
            print('Item ' + str(i))
            if len(index_alpha) > 0:
                print_lines(index_alpha[i], lines, '  ')
            for b in item:
                print_lines(index_beta[b], lines, '  ')
        if len(index_kappa) > 0:
            print('Rating scale step parameters')
            print_lines(index_kappa, lines, '  ')
        print('Ability distribution')
        print_lines(index_lambda, lines, '  ')
        print_lines(index_sigma, lines, '  ')
        print_lines(index_bottom, lines)
