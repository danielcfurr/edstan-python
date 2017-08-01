# For basic functionality
import pystan, numpy, patsy, os

# For print functionality
import re, io, contextlib

class EdstanData:
    """Data arranged for compatibility with the edstan models.

    Input formats:
        Item response data may be provided in 'wide' or 'long' form. 
        For wide-form data, scored responses are contained in a response matrix in 
        which the rows 
        represent persons and the columns represent items. The response matrix is
        passed to the `response_matrix` argument, and the arguments `item_id`,
        `person_id`, and `y` are not used. The response matrix should be a 
        two-dimensional numpy array or an object that is convertible to the same.
        
        Long-form data consist of a scored response vector along with two vectors
        serving as person and item indicators. In this case, arguments `y`, 
        `item_id`, and `person_id` are provided. Each should be a one-dimensional
        numpy array or  an object that is convertible to the same. Further, 
        `item_id` and `person_id` must contain consective integers with the
        lowest values equalling one (not zero).
    
        Whichever format the original data are in, the scored responses should 
        have a lowest value of zero for any given item and should not contain 
        missing response categories. (For example, if an item has three response
        categories, then the scores 0, 1, and 2 should be found in the data at least
        once.) These caveats may not apply to rating scale models. Also, the 
        scored responses may contain missing values.
        
    Latent regression:
        A latent regression of person ability on covariates may optionally be 
        included in one of two ways. First, a design matrix for the latent 
        regression may be supplied directly to the `person_data` argument while
        omitting the `formula` argument. This matrix should be a two-dimensional
        numpy array or an object that is convertible to the same. The first column 
        must have all elements equal to one to serve as the model intercept. 
        
        Alternatively, both `person_data` and `formula` may be specified for the
        latent regression. In this 
        case, `person_data` should be a data structure compatible with 
        `patsy.dmatrix()`, such as a pandas data frame, and `formula` should
        be a string formula also compatible with `patsy.dmatrix()`. In the event
        that both `person_data` and `formula` are omitted, the latent regression
        is restricted to the model intercept only.

    Args:
        response_matrix (optional): 
            An object containing scored response that 
            can be converted to a two-dimensional numpy array. The rows 
            represent persons and the columns represent items. If provided, 
            `item_id`, `person_id`, and `y` are ignored.
        item_id (optional): 
            An object indexing the items that can be 
            converted to a one-dimensional numpy array. Used instead of
            `response_matrix` and must be of same length as `y`.
        person_id (optional): 
            An object indexing the persons that can be 
            converted to a one-dimensional numpy array. Used instead of
            `response_matrix` and must be of same length as `y`.
        y (optional): 
            An object containing scored responses that can be 
            converted to a one-dimensional numpy array. Used along with
            `item_id` and `person_id`.
        person_data (optional): 
            An object containing person covariates that 
            is compatible with `patsy.`. Must have
            a number of rows equal to either the number of rows in
            `response_matrix` or the length of `y`. If provided without 
            `formula`, the first column must have all elements equal to one to
            serve as the model intercept.
        formula (optional): 
            A string that is a patsy-style formula applied to
            `person_data`.

    Attributes:
        datalist: The data as arrange for the edstan models.
        
    Examples:
        Use pandas to import the spelling data as a data frame with one
        row per person. Four columns correspond to item responses, and a
        fifth column is a dummy variable for whether the respondent is male.

        >>> import pandas
        >>> spelling = pandas.read_csv('spelling.csv')
        >>> words = ['infidelity', 'panoramic', 'succumb', 'girder']
        
        Get EdstanData objects for fitting models with and without the
        latent regression on person covariates.
        
        >>> ed_1 = edstan.EdstanData(response_matrix = spelling[words])
        >>> ed_2 = edstan.EdstanData(response_matrix = spelling[words],
        >>>                          person_data = spelling['male'],
        >>>                          formula = '~male')
        
        Use pandas to import the verbal aggression data as a data frame with
        one row per response. The columns include the scored response,
        item indicator, person indicator, and person-related covariates.
        
        >>> import pandas
        >>> aggression = pandas.read_csv('aggression.csv')
        
        Get EdstanData objects for fitting models with and without the
        latent regression on person covariates.
        
        >>> ed_3 = edstan.EdstanData(item_id = aggression['item'], 
        >>>                          person_id = aggression['person'],
        >>>                          y = aggression['poly'])
        >>> ed_4 = edstan.EdstanData(item_id = aggression['item'], 
        >>>                          person_id = aggression['person'],
        >>>                          y = aggression['poly'],
        >>>                          person_data = aggression[['male', 'anger']],
        >>>                          formula = '~ male + anger')
    """
    
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
    
    def __str__(self):
        
        #!! Do something with this?
       pass
    
    def max_score_item(self):
        """Provides the maximum score for each item.
            
        Returns:
            A list of maximum scores.
        """
        
        max_score = [0] * self.datalist['I']
        for n in range(self.datalist['N']):
            i = self.datalist['ii'][n] - 1
            max_score[i] = max(self.datalist['y'][n], max_score[i])
        return max_score

    def beta_key(self, rating_scale=False):
        """Provides a map indicating which item difficulties correspond to
        which items.
        
        This is mainly useful for the (generalized) partial credit model, 
        where items will have more than one dificulty parameter.
        
        Args:
            rating_scale (optional): 
                Whether a map is wanted for the (generalized) rating scale
                model. Default is false.
            
        Returns:
            A list of ranges. Each range provides the index of betas for a
            given item.
        """
        
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
        """Provides each respondent's raw score.
            
        Returns:
            A list of raw scores.
        """
        
        raw_scores = [0] * self.datalist['J']
        for n in range(self.datalist['N']):
            j = self.datalist['jj'][n] - 1
            raw_scores[j] = raw_scores[j] + self.datalist['y'][n]
        return raw_scores
    
    def fit_model(self, model, output=True, **kwargs):
        """Fits a Stan model to an EdstanData object.

        The available models are the Rasch, rating scale, partical credit,
        two-parameter logistic, generalized rating scale, and generalized
        partial credit models.

        Args:
            model: 
                The choice of model ('rasch', 'rsm', 'pcm', '2pl', 'grsm',
                or 'gpcm').
            Output (optional): 
                Whether to print output after fitting the model. Default
                is True.
            **kwargs (optional): 
                Additional parameters passed to `pystan.stan()`. In
                particular, choices for `iter` and `chains` should
                be supplied.
            
        Returns:
            A `pystan.StanFit4model` instance.
            
        Examples:
            Import the spelling data, create an `EdstanData` instance, and fit
            the Rasch model.
    
            >>> import pandas
            >>> spelling = pandas.read_csv('spelling.csv')
            >>> words = ['infidelity', 'panoramic', 'succumb', 'girder']
            >>> ed_1 = edstan.EdstanData(response_matrix = spelling[words])
            >>> ed_1.fit_model('rasch', iter=200, chains=4)
        """
        
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
        """Prints output for a Stan model.

        The benefit of using this over the default `print` for a 
        `pystan.StanFit4model` instance is that summaries of item parameter
        posteriors will be grouped by item, and also that the displayed
        parameter summaries are filtered to include only those of likely
        interest. This output is provided by default when a model is fit.

        Args:
            fit: 
                A `pystan.StanFit4model` instance generated by 
                `EdstanData.fit_model()`.
        """        
        
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
