import jmaestre as jm
import unittest

class SimpleLinearRegressionCases(unittest.TestCase):
    known_values = (
        (((-10,0,10), (-10,0,10)), (0,1)),
        (((10,0,-10), (-10,0,10)), (0,-1))
    )                     #(((-10,0,10), (10,0,-10)), (0,-1)))

    def test_SimpleLinearRegression_fit(self):
        '''fit method should give known result with known input'''
        simple_linear_regressor = jm.SimpleLinearRegressor()
        for data, coefs in self.known_values:
            simple_linear_regressor.fit(X = data[0], Y = data[1])
            self.assertEqual(simple_linear_regressor.parameters['intercept'],coefs[0])
            self.assertEqual(simple_linear_regressor.parameters['slope'],coefs[1])
            
            
if __name__ == '__main__':
    unittest.main()