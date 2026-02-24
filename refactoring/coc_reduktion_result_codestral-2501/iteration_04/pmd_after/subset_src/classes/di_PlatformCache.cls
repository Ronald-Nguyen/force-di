public with sharing class di_PlatformCache
{
    public static final Integer NUM_OF_SECS_IN24HR =  86400;

    @TestVisible
    private static final String BINDING_KEY = 'bindingKeyIndex';
    @TestVisible
    private static di_PlatformCache instance;
    @TestVisible
    private Map<String, Map<Schema.SObjectType, Set<String> > > cacheKeyIndexMap = new Map<String, Map<Schema.SObjectType, Set<String> > >();
    @TestVisible
    private static Map<String, String> generatedKeyNames = new Map<String, String>();

    @TestVisible
    private static Boolean DEBUGGING = false;

    private di_PlatformCache() {}

    public static di_PlatformCache getInstance() {
        if (instance == null) {
            instance = new di_PlatformCache();
        }

        return instance;
    }

    public static Boolean isStoringBindingInPlatformCache() {
        return getConfig().UsePlatformCacheToStoreBindings__c == null ?
            false :
            (getConfig().UsePlatformCacheToStoreBindings__c && getPartition() != null);
    }

    public Map<String, Map<Schema.SObjectType, Set<String>>> getCacheKeyIndexMap() {
        if (cacheKeyIndexMap.isEmpty()) {
            loadCacheKeyIndexMapFromPartition();
        }

        return cacheKeyIndexMap;
    }

    private void loadCacheKeyIndexMapFromPartition() {
        try {
            Cache.OrgPartition orgPartition = getPartition();
            if (orgPartition != null) {
                cacheKeyIndexMap = (Map<String, Map<Schema.SObjectType, Set<String> > >) orgPartition.get(getKeyIndexName());
            }
        } catch (Cache.Org.OrgCacheException ex) {
            cacheKeyIndexMap = null;
        }

        if (cacheKeyIndexMap == null) {
            cacheKeyIndexMap = new Map<String, Map<Schema.SObjectType, Set<String> > >();
        }
    }

    public Boolean addBindingToPlatformCache(di_Binding binding) {
        if (!isStoringBindingInPlatformCache()) {
            return false;
        }

        Cache.OrgPartition orgPartition = getPartition();
        if (orgPartition == null) {
            return false;
        }

        String theKeyName = getKeyName(binding);
        List<di_Binding> workingBindings = getExistingBindingsForKey(orgPartition, theKeyName);

        workingBindings.add(binding);
        log('Adding binding for hash => ' + theKeyName + ' && developerName => ' + binding.developerName + ' && object => ' + binding.bindingObject);

        try {
            orgPartition.put(theKeyName, workingBindings, getPartitionTTL(), Cache.Visibility.ALL, false);
        } catch (cache.Org.OrgCacheException oce) {
            log(oce);
        }

        addBindingToKeyIndex(binding);
        return true;
    }

    private List<di_Binding> getExistingBindingsForKey(Cache.OrgPartition orgPartition, String theKeyName) {
        List<di_Binding> workingBindings = new List<di_Binding>();

        if (orgPartition.contains(theKeyName)) {
            workingBindings.addAll((List<di_Binding>) orgPartition.get(theKeyName));
        }

        return workingBindings;
    }

    public list<di_Binding> retrieveBindings(String developerName, Schema.SObjectType bindingSObjectType) {
        list<di_Binding> bindings = new list<di_Binding>();

        if (!shouldRetrieveFromCache(developerName, bindingSObjectType)) {
            return bindings;
        }

        log('Retrieving from Cache Key => ' + developerName + ' && Cache Key Index => ' + bindingSObjectType);

        Cache.OrgPartition orgPartition = getPartition();
        Map<Schema.SObjectType, Set<String> > keyIndexBySObjectTypeMap = getCacheKeyIndexMap().get(developerName.toLowerCase().trim());

        if (keyIndexBySObjectTypeMap == null || orgPartition == null) {
            return bindings;
        }

        Set<String> cacheKeys = keyIndexBySObjectTypeMap.get(bindingSObjectType);
        if (cacheKeys == null) {
            return bindings;
        }

        return getBindingsFromCacheKeys(orgPartition, cacheKeys);
    }

    private Boolean shouldRetrieveFromCache(String developerName, Schema.SObjectType bindingSObjectType) {
        return isStoringBindingInPlatformCache() &&
               string.isNotBlank(developerName) &&
               bindingSObjectType != null;
    }

    private list<di_Binding> getBindingsFromCacheKeys(Cache.OrgPartition orgPartition, Set<String> cacheKeys) {
        list<di_Binding> bindings = new list<di_Binding>();
        Object cachedObject = null;

        for (String cacheKey : cacheKeys) {
            cachedObject = orgPartition.get(cacheKey);
            if (cachedObject != null) {
                bindings.addAll((List<di_Binding>) cachedObject);
            }
        }

        return bindings;
    }

    public static void clearCachedBindings() {
        Set<String> keys = getPartitionKeys();

        if (keys == null) {
            return;
        }

        String partitionKey = getPartitionName();
        Cache.OrgPartition partition = getPartition();

        for (String key : partition.getKeys()) {
            try {
                partition.remove(key);
            } catch (Exception ex) {
                log('XX]> Unable to remove Platform Cache partition [' + partitionKey + '] key [' + key + ']');
            }
        }
    }

    public static Set<String> getPartitionKeys() {
        Set<String> keys = null;
        Cache.OrgPartition partition = getPartition();

        if (partition != null) {
            keys = partition.getKeys();
        }

        return keys;
    }

    @TestVisible
    private static di_Configurations__c getConfig() {
        di_Configurations__c config = di_Configurations__c.getInstance();

        if (config == null) {
            config = createNewConfig();
        }

        return config;
    }

    private static di_Configurations__c createNewConfig() {
        di_Configurations__c config = new di_Configurations__c();
        config.SetupOwnerId = ConnectApi.Organization.getSettings().orgId;

        if (di_Configurations__c.SObjectType.getDescribe().isCreateable()) {
            insert config;
        }

        return config;
    }

    @TestVisible
    private static String getPartitionName() {
        return getConfig().OrgCachePartitionName__c;
    }

    @TestVisible
    private Integer getPartitionTTL() {
        return NUM_OF_SECS_IN24HR;
    }

    @TestVisible
    private static Cache.OrgPartition getPartition() {
        Cache.OrgPartition result = null;

        try {
            result = Cache.Org.getPartition(getPartitionName());
        } catch (Exception excp) {
            log('ERROR: Is there Cache? Is the Cache Partition enabled : Exception:' + excp);
        }

        return result;
    }

    @TestVisible
    private boolean pushCacheKeyIndexMapToCache() {
        Cache.OrgPartition orgPartition = getPartition();

        if (orgPartition != null) {
            orgPartition.put(getKeyIndexName(), this.cacheKeyIndexMap, getPartitionTTL(), Cache.Visibility.ALL, false);
        }

        return orgPartition != null;
    }

    @TestVisible
    private void addBindingToKeyIndex(di_Binding binding) {
        if (binding == null) {
            return;
        }

        String workingDeveloperName = binding.developerName.toLowerCase().trim();

        initializeKeyIndexForDeveloperName(workingDeveloperName);
        initializeKeyIndexForBindingObject(workingDeveloperName, binding);

        getCacheKeyIndexMap().get(workingDeveloperName).get(binding.bindingObject).add(getKeyName(binding));
        pushCacheKeyIndexMapToCache();
    }

    private void initializeKeyIndexForDeveloperName(String workingDeveloperName) {
        if (!getCacheKeyIndexMap().containsKey(workingDeveloperName)) {
            getCacheKeyIndexMap().put(workingDeveloperName, new Map<Schema.SObjectType, Set<String>>());
        }
    }

    private void initializeKeyIndexForBindingObject(String workingDeveloperName, di_Binding binding) {
        if (!getCacheKeyIndexMap().get(workingDeveloperName).containsKey(binding.bindingObject)) {
            getCacheKeyIndexMap().get(workingDeveloperName).put(binding.bindingObject, new Set<String>());
        }
    }

    @TestVisible
    private String constructKeyName(Schema.SObjectType bindingSObjectType, String developerName) {
        if (String.isBlank(developerName)) {
            return null;
        }

        String key = buildKeyString(bindingSObjectType, developerName);

        if (generatedKeyNames.containsKey(key)) {
            return generatedKeyNames.get(key);
        }

        String hash = generateHashForKey(key, developerName, bindingSObjectType);
        generatedKeyNames.put(key, hash);
        return hash;
    }

    private String buildKeyString(Schema.SObjectType bindingSObjectType, String developerName) {
        String objectName = (bindingSObjectType != null) ?
            bindingSObjectType.getDescribe().getName().toLowerCase().replaceAll('__','') :
            '';

        return objectName + (String.isBlank(developerName) ? '' : developerName.toLowerCase().trim());
    }

    private String generateHashForKey(String key, String developerName, Schema.SObjectType bindingSObjectType) {
        String hash = String.valueOf(Math.abs((key).hashcode()));
        log('Creating Hash For => ' + developerName + ' && ' + bindingSObjectType + ' := ' + hash);
        return hash;
    }

    @TestVisible
    private String getKeyName(String developerName, Schema.SObjectType bindingSObjectType) {
        return constructKeyName(bindingSObjectType, developerName);
    }

    @TestVisible
    private String getKeyName(di_Binding binding) {
        return constructKeyName(binding.bindingObject, binding.developerName);
    }

    @TestVisible
    private String getKeyIndexName() {
        return di_PlatformCache.BINDING_KEY;
    }

    @TestVisible
    private static Boolean log(Exception excp) {
        Boolean before = DEBUGGING, result;

        DEBUGGING = true;
        result = log('Exception: ' + excp.getMessage());
        DEBUGGING = before;
        return result;
    }

    @TestVisible
    private static Boolean log(String message) {
        Boolean doLog = DEBUGGING && string.isNotBlank(message);

        if (doLog) {
            system.debug('+++ di_PlatformCache.message: ' + message);
        }

        return doLog;
    }
}