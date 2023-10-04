import pickle

from django.shortcuts import render
from django.views import View

with open('trained_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


class CustomerView(View):
    def get(self, request):
        return render(request, 'home.html')

    def post(self, request):

        credit_score = request.POST.get('creditscore')
        age = request.POST.get('age')
        tenure = request.POST.get('tenure')
        balance = request.POST.get('balance')

        no_of_products = request.POST.get('num-of-products')
        has_credit_card = request.POST.get('has-credit-card')
        is_active_member = request.POST.get('is-active-member')
        salary = request.POST.get('salary')
        gender = request.POST.get('gender')
        geography = request.POST.get('geography')

        if has_credit_card == 'yes':
            has_credit_card = 1
        else:
            has_credit_card = 0

        if is_active_member == 'yes':
            is_active_member = 1
        else:
            is_active_member = 0

        if gender == 'male':
            gender = 1
        else:
            gender = 0

        if geography == 'germany':
            Geography_Germany = 1
            Geography_Spain = 0
            Geography_France = 0
        elif geography == 'spain':
            Geography_Germany = 0
            Geography_Spain = 1
            Geography_France = 0
        elif geography == 'france':
            Geography_Germany = 0
            Geography_Spain = 0
            Geography_France = 1

        y_pred = loaded_model.predict([[
            credit_score, age, tenure, balance, no_of_products,
            has_credit_card, is_active_member, salary, Geography_Germany,
            Geography_Spain, gender
        ]])[0]

        if y_pred == 1:
            return render(request, 'home.html',
                          {'result': 'Customer Will Leave the Bank'})
        else:
            return render(request, 'home.html',
                          {'result': 'Customer Will Not Leave the Bank'})
