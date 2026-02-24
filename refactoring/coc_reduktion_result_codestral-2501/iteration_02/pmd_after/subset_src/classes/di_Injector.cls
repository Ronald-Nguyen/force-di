public class di_Injector {
    @testVisible
    private static List<di_BindingConfigWrapper> mock_BindingConfigurationWrappersOuter = null;

    public static final di_Injector Org =
        new di_Injector(
            new List<di_Module> {
        new CustomMetadataModule()
    });

    @TestVisible
    public di_Binding.Resolver Bindings {get; private set;}

    public di_Injector(di_Module module) {
        this(new List<di_Module> { module });
    }

    public di_Injector(List<di_Module> modules) {
        Bindings = new di_Binding.Resolver(modules);
    }

    public Object getInstance(Type developerNameByType) {
        return getInstance(developerNameByType?.getName()?.toLowerCase(), null);
    }

    public Object getInstance(Type developerNameByType, Object params) {
        return getInstance(developerNameByType?.getName()?.toLowerCase(), params);
    }

    public Object getInstance(String developerName) {
        return getInstance(developerName, null);
    }

    public Object getInstance(String developerName, Object params) {
        if (String.isBlank(developerName)) {
            throw new InjectorException('Request for Binding cannot take "developerName" parameter of null');
        }

        List<di_Binding> bindingsFound = getBindingsByName(developerName);
        if (bindingsFound == null || bindingsFound.isEmpty()) {
            throw new InjectorException('Binding for "' + developerName + '" not found');
        }
        return bindingsFound[0].getInstance(params);
    }

    private List<di_Binding> getBindingsByName(String developerName) {
        return this.Bindings.byName(developerName.toLowerCase().trim()).get();
    }

    public Object getInstance(Type developerNameByType, Schema.SObjectType bindingSObjectType) {
        return getInstance(developerNameByType, bindingSObjectType, null);
    }

    public Object getInstance(Type developerNameByType, Schema.SObjectType bindingSObjectType, Object params) {
        return getInstance(developerNameByType == null ? null : developerNameByType.getName().toLowerCase(), bindingSObjectType, params);
    }

    public Object getInstance(String developerName, Schema.SObjectType bindingSObjectType, Object params) {
        validateParameters(developerName, bindingSObjectType);

        List<di_Binding> bindingsFound = getBindingsByNameAndObject(developerName, bindingSObjectType);

        if (bindingsFound == null || bindingsFound.isEmpty() || bindingsFound[0] == null) {
            throw new InjectorException('Binding for "' + developerName + '" and SObjectType "' + bindingSObjectType + '" not found');
        }
        return bindingsFound[0].getInstance(params);
    }

    private void validateParameters(String developerName, Schema.SObjectType bindingSObjectType) {
        if (String.isBlank(developerName)) {
            throw new InjectorException('Request for Binding cannot take "developerName" parameter of null');
        }

        if (bindingSObjectType == null) {
            throw new InjectorException('Request for Binding cannot take "bindingSObjectType" parameter of null');
        }
    }

    private List<di_Binding> getBindingsByNameAndObject(String developerName, Schema.SObjectType bindingSObjectType) {
        return this.Bindings.bySObject(bindingSObjectType)
            .byName(developerName.toLowerCase().trim())
            .get();
    }

    @testVisible
    private class CustomMetadataModule extends di_Module {

        private string bindingObjectApiName = null;

        @testVisible
        private List<di_BindingConfigWrapper> getDIBinding(){
            List<di_BindingConfigWrapper> recordList = new List<di_BindingConfigWrapper>();

            if(di_Injector.mock_BindingConfigurationWrappersOuter != null) {
                recordList = di_Injector.mock_BindingConfigurationWrappersOuter;
            } else {
                List<di_Binding__mdt> bindingMDTRec = [SELECT QualifiedAPIName
                                                            , DeveloperName
                                                            , BindingName__c
                                                            , NamespacePrefix
                                                            , Type__c
                                                            , To__c
                                                            , BindingObject__c
                                                            , BindingObject__r.QualifiedApiName
                                                            , BindingObjectAlternate__c
                                                            , BindingSequence__c
                                                         FROM di_Binding__mdt];
                for(di_Binding__mdt records :bindingMDTRec) {
                    recordList.add(new di_BindingConfigWrapper(records));
                }
            }
            return recordList;
        }

        public override void configure() {
            for(di_BindingConfigWrapper bindingConfig : getDIBinding()) {
                configureBinding(bindingConfig);
            }
        }

        private void configureBinding(di_BindingConfigWrapper bindingConfig) {
            bind(bindingConfig.DeveloperName);
            type(bindingConfig.Type);

            if(shouldBindToObject(bindingConfig)) {
                bindToObject(bindingConfig);
            }

            if(bindingConfig.BindingSequence != null) {
                sequence(Integer.valueOf(bindingConfig.BindingSequence));
            }

            data(bindingConfig);
            to(bindingConfig.To);
        }

        private Boolean shouldBindToObject(di_BindingConfigWrapper bindingConfig) {
            return String.isNotBlank(bindingConfig.BindingObject) ||
                   String.isNotBlank(bindingConfig.BindingObjectAlternate);
        }

        private void bindToObject(di_BindingConfigWrapper bindingConfig) {
            bindingObjectApiName = getBindingObjectApiName(bindingConfig);
            Schema.DescribeSobjectResult[] results = Schema.describeSObjects(new String[] { bindingObjectApiName });

            if(results.size() != 1) {
                throw new InjectorException('Failed to find SObject ' + bindingObjectApiName + ' referenced by binding ' + bindingConfig.DeveloperName);
            }

            bind(results[0].getSObjectType());
        }

        private String getBindingObjectApiName(di_BindingConfigWrapper bindingConfig) {
            return String.isNotBlank(bindingConfig.BindingObject)
                ? bindingConfig.bindingObjectQualifiedApiName.toLowerCase().trim()
                : bindingConfig.BindingObjectAlternate.toLowerCase().trim();
        }
    }

    public class InjectorException extends Exception {}
}