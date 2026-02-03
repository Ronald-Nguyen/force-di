public with sharing class di_InjectorController {

    public ApexPages.StandardController StandardController {get;set;}

    public di_InjectorController(ApexPages.StandardController StandardController) {
        this.StandardController = StandardController;
    }

    public di_InjectorController getThis() {
        return this;
    }

    @AuraEnabled
    public static di_Binding getBinding(String bindingName) {
        return di_Injector.Org.Bindings.byName(bindingName).get()[0];
    }
}