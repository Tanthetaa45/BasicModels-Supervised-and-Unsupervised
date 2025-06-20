import re

def parse_polynomial(polynomial):
    # Remove spaces and split by terms
    polynomial = polynomial.replace(' ', '')
    terms = re.findall(r'[+-]?\d*x\^?\d*', polynomial)
    
    parsed_terms = []
    for term in terms:
        # Separate coefficient and power
        if 'x^' in term:
            coefficient, power = term.split('x^')
        elif 'x' in term:
            coefficient = term.split('x')[0]
            power = '1'
        else:
            coefficient = term
            power = '0'
        
        # Handle cases with implicit coefficients and powers
        if coefficient == '+' or coefficient == '':
            coefficient = '1'
        elif coefficient == '-':
            coefficient = '-1'
        
        parsed_terms.append((int(coefficient), int(power)))
    
    return parsed_terms

def derivative(parsed_polynomial):
    derived_terms = []
    for coefficient, power in parsed_polynomial:
        if power != 0:
            derived_coefficient = coefficient * power
            derived_power = power - 1
            derived_terms.append((derived_coefficient, derived_power))
    
    return derived_terms

def format_polynomial(terms):
    formatted_terms = []
    for coefficient, power in terms:
        if power == 0:
            formatted_terms.append(f'{coefficient}')
        elif power == 1:
            formatted_terms.append(f'{coefficient}x')
        else:
            formatted_terms.append(f'{coefficient}x^{power}')
    
    return ' + '.join(formatted_terms).replace('+-', '- ')

def main():
    polynomial = input("Enter the polynomial (e.g., 3x^3 + 2x^2 - 5x + 7): ")
    parsed_polynomial = parse_polynomial(polynomial)
    derived_polynomial = derivative(parsed_polynomial)
    formatted_derivative = format_polynomial(derived_polynomial)
    print(f"The derivative of the polynomial is: {formatted_derivative}")

if __name__ == "__main__":
    main()
